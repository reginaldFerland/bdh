# Task 0008: Add Distributed Training Support

## Priority
**Medium** - Important for scaling to larger models and datasets

## Purpose
Implement multi-GPU training using PyTorch's Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP) to enable training larger BDH models across multiple GPUs or nodes. This significantly speeds up training and enables models that don't fit on a single GPU.

## Current State
- Training only supports single GPU
- No distributed training capabilities
- No gradient accumulation
- Limited to models that fit on one GPU
- No multi-node training support

## Expected Outcome
After implementing this task, the project should have:
1. DDP (Distributed Data Parallel) support for multi-GPU training
2. FSDP (Fully Sharded Data Parallel) for very large models
3. Gradient accumulation for effective larger batch sizes
4. Automatic device placement and synchronization
5. Support for multi-node training
6. Proper handling of model compilation with distributed training
7. Configuration options for distributed settings

## Detailed Requirements

### 1. DDP (Distributed Data Parallel)

Best for models that fit on a single GPU:
- Replicates model on each GPU
- Splits batch across GPUs
- Synchronizes gradients after backward pass
- Scales well up to 8-16 GPUs

### 2. FSDP (Fully Sharded Data Parallel)

For models too large for single GPU:
- Shards model parameters across GPUs
- Shards optimizer states
- Reduces memory per GPU
- Enables training much larger models

### 3. Gradient Accumulation

Simulate larger batches without more memory:
- Accumulate gradients over N steps
- Update weights after N accumulations
- Effective batch size = batch_size * accumulation_steps * num_gpus

### 4. Mixed Precision Training

Already supported, ensure compatibility with DDP/FSDP:
- Use torch.cuda.amp.autocast()
- GradScaler for float16
- BFloat16 doesn't need scaling

## Implementation Steps

### Step 1: Create Distributed Training Utilities

Create `distributed.py`:
```python
"""Distributed training utilities for BDH."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_world_size():
    """Get world size (number of processes)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    """Get rank (process ID)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0


def wrap_model_ddp(model, device):
    """Wrap model with DistributedDataParallel."""
    if get_world_size() > 1:
        model = DDP(model, device_ids=[device])
    return model


def wrap_model_fsdp(model, auto_wrap_policy=None):
    """Wrap model with FullyShardedDataParallel."""
    if get_world_size() > 1:
        if auto_wrap_policy is None:
            # Default: wrap layers with > 100M parameters
            auto_wrap_policy = size_based_auto_wrap_policy(
                min_num_params=100_000_000
            )
        
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=None,  # Handle externally
            device_id=torch.cuda.current_device(),
        )
    return model


class DistributedSampler:
    """Simple distributed sampler for data loading."""
    
    def __init__(self, dataset_size, batch_size, rank=None, world_size=None):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.rank = rank if rank is not None else get_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
        
        # Calculate per-rank samples
        self.num_samples = dataset_size // self.world_size
        self.start_idx = self.rank * self.num_samples
        self.end_idx = self.start_idx + self.num_samples
    
    def get_indices(self):
        """Get indices for this rank."""
        return range(self.start_idx, self.end_idx)
```

### Step 2: Update Training Script for DDP

Modify `train.py`:
```python
from distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    wrap_model_ddp,
    get_world_size,
)

# Setup distributed
rank, world_size, local_rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

print(f"Rank {rank}/{world_size} using device {device}")

# Create model
model = bdh.BDH(BDH_CONFIG).to(device)

# Wrap with DDP (don't compile before wrapping)
if world_size > 1:
    model = wrap_model_ddp(model, local_rank)

# Now compile if requested
if args.compile:
    model = torch.compile(model)

# Training loop
for step in range(start_step, MAX_ITERS):
    # Forward/backward as usual
    with ctx:
        logits, loss = model(x, y)
    
    # Scale loss by world size for gradient accumulation
    loss = loss / args.gradient_accumulation_steps
    
    scaler.scale(loss).backward()
    
    # Accumulate gradients
    if (step + 1) % args.gradient_accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Only log/save on main process
    if is_main_process():
        if step % LOG_FREQ == 0:
            print(f"Step {step}: loss = {loss.item()}")
        
        if step % SAVE_FREQ == 0:
            checkpoint_manager.save_checkpoint(...)

# Cleanup
cleanup_distributed()
```

### Step 3: Add Gradient Accumulation

```python
# Add to argparse
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                   help='Number of steps to accumulate gradients')

# In training loop
accumulated_loss = 0
for step in range(start_step, MAX_ITERS):
    with ctx:
        logits, loss = model(x, y)
    
    # Scale loss
    loss = loss / args.gradient_accumulation_steps
    accumulated_loss += loss.item()
    
    # Backward
    scaler.scale(loss).backward()
    
    # Update weights every N steps
    if (step + 1) % args.gradient_accumulation_steps == 0:
        # Optional: gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Log accumulated loss
        if is_main_process() and step % LOG_FREQ == 0:
            print(f"Step {step}: loss = {accumulated_loss:.4f}")
        accumulated_loss = 0
    
    # Get next batch
    x, y = get_batch("train")
```

### Step 4: Add FSDP Support

For very large models:
```python
from distributed import wrap_model_fsdp
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)

# FSDP configuration
if args.use_fsdp:
    # Mixed precision policy
    mp_policy = None
    if dtype == "float16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif dtype == "bfloat16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    # Wrap model
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
    )
```

### Step 5: Create Launch Script

Create `scripts/train_distributed.sh`:
```bash
#!/bin/bash

# Single node, multiple GPUs
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    train.py \
    --dataset_name wikitext \
    --gradient_accumulation_steps 4 \
    --batch_size 8

# Multiple nodes (e.g., 2 nodes with 4 GPUs each)
# On master node:
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train.py \
    --dataset_name wikitext \
    --batch_size 8

# On worker node:
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train.py \
    --dataset_name wikitext \
    --batch_size 8
```

### Step 6: Handle Checkpointing with DDP/FSDP

```python
def save_checkpoint_distributed(model, optimizer, scaler, step, checkpoint_manager):
    """Save checkpoint in distributed setting."""
    if is_main_process():
        # Unwrap DDP/FSDP model
        model_to_save = model.module if hasattr(model, 'module') else model
        
        checkpoint_manager.save_checkpoint(
            state={
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'step': step,
                'config': vars(args),
            },
            step=step,
        )
    
    # Wait for all processes
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint_distributed(model, optimizer, scaler, checkpoint_path):
    """Load checkpoint in distributed setting."""
    # Load on main process first
    map_location = {'cuda:0': f'cuda:{get_rank()}'}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load into model
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer and scaler
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['step']
```

### Step 7: Add Command-Line Arguments

```python
parser.add_argument('--distributed', action='store_true',
                   help='Enable distributed training')
parser.add_argument('--use_ddp', action='store_true',
                   help='Use DistributedDataParallel')
parser.add_argument('--use_fsdp', action='store_true',
                   help='Use FullyShardedDataParallel')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                   help='Gradient accumulation steps')
parser.add_argument('--grad_clip', type=float, default=1.0,
                   help='Gradient clipping value (0 to disable)')
```

## Testing Plan

1. **Test Single GPU (Baseline)**
   - Train without distributed
   - Record training time and memory
   - Verify convergence

2. **Test DDP (Multi-GPU, Same Node)**
   - Train with 2, 4, 8 GPUs
   - Verify speedup scales linearly
   - Check memory usage per GPU
   - Verify final loss matches single-GPU

3. **Test Gradient Accumulation**
   - Train with accumulation_steps=4
   - Verify effective batch size increase
   - Check convergence matches larger batch

4. **Test FSDP (Large Model)**
   - Create model too large for single GPU
   - Train with FSDP on multiple GPUs
   - Verify training works
   - Check memory distribution

5. **Test Checkpoint Saving/Loading**
   - Save checkpoint during distributed training
   - Resume from checkpoint
   - Verify training continues correctly

6. **Test Multi-Node**
   - Train on 2 nodes with 4 GPUs each
   - Verify communication works
   - Check speedup vs single node

7. **Test with torch.compile()**
   - Combine DDP with torch.compile()
   - Verify compatibility
   - Check performance improvement

## Code Example

```bash
# Single GPU
python train.py --dataset_name wikitext

# Multi-GPU with DDP
torchrun --nproc_per_node=4 train.py \
    --dataset_name wikitext \
    --use_ddp \
    --batch_size 8

# With gradient accumulation (effective batch_size=32)
torchrun --nproc_per_node=4 train.py \
    --dataset_name wikitext \
    --use_ddp \
    --batch_size 8 \
    --gradient_accumulation_steps 4

# FSDP for large model
torchrun --nproc_per_node=8 train.py \
    --dataset_name wikitext \
    --use_fsdp \
    --batch_size 4

# Multi-node (on master node)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train.py \
    --dataset_name wikitext \
    --use_ddp
```

## Copilot Implementation Prompt

```
Implement distributed training support for BDH:

1. Create distributed.py with utilities:
   - setup_distributed(): Initialize torch.distributed, return rank, world_size, local_rank
   - cleanup_distributed(): Destroy process group
   - get_rank(), get_world_size(), is_main_process(): Helper functions
   - wrap_model_ddp(model, device): Wrap model with DDP
   - wrap_model_fsdp(model): Wrap model with FSDP

2. Update train.py to support distributed training:
   - Call setup_distributed() at start
   - Set device to cuda:{local_rank}
   - Wrap model with DDP or FSDP based on args
   - Only log and save checkpoints on main process (is_main_process())
   - Call cleanup_distributed() at end

3. Add gradient accumulation:
   - Add --gradient_accumulation_steps argument
   - Scale loss by 1/accumulation_steps
   - Only call optimizer.step() every N steps
   - Accumulate loss for logging

4. Handle checkpointing:
   - Unwrap DDP/FSDP model before saving: model.module if hasattr(model, 'module') else model
   - Only save on main process
   - Add dist.barrier() after saving
   - Load with correct map_location for each rank

5. Add command-line arguments:
   - --use_ddp: Use DistributedDataParallel
   - --use_fsdp: Use FullyShardedDataParallel  
   - --gradient_accumulation_steps: Number of accumulation steps
   - --grad_clip: Gradient clipping value

6. DDP implementation:
   - Initialize process group with backend='nccl'
   - Wrap model after moving to device, before compiling
   - Use DDP(model, device_ids=[local_rank])

7. FSDP implementation:
   - Import from torch.distributed.fsdp
   - Use ShardingStrategy.FULL_SHARD
   - Configure MixedPrecision based on dtype
   - Set device_id=torch.cuda.current_device()

8. Launch with torchrun:
   - Use torchrun instead of python for multi-GPU
   - torchrun sets RANK, WORLD_SIZE, LOCAL_RANK env variables
   - Format: torchrun --nproc_per_node=<num_gpus> train.py <args>

Current training loop is in train.py lines 102-126. Integrate distributed training without breaking single-GPU mode.

Environment variables set by torchrun:
- RANK: Global process rank
- WORLD_SIZE: Total number of processes
- LOCAL_RANK: Rank within node
- MASTER_ADDR: Master node address (for multi-node)
- MASTER_PORT: Master node port (for multi-node)
```

## Files to Modify/Create
- **Create**: `distributed.py` - Distributed training utilities
- **Modify**: `train.py` - Add distributed training support
- **Create**: `scripts/train_distributed.sh` - Launch script examples

## Dependencies
- PyTorch >= 2.0 (for FSDP improvements)
- No new external dependencies
- Requires NCCL for multi-GPU (comes with PyTorch)

## Success Criteria
- [ ] Can train on single GPU (backward compatible)
- [ ] DDP works on multiple GPUs
- [ ] Training speeds up with more GPUs
- [ ] FSDP enables larger models
- [ ] Gradient accumulation works correctly
- [ ] Checkpointing works with distributed
- [ ] Multi-node training works
- [ ] torch.compile() is compatible
- [ ] Documentation includes distributed examples

## Related Tasks
- **Task 0003**: Checkpoint Management (needs distributed-aware saving)
- **Task 0004**: Training Monitoring (log only on main process)
- **Task 0009**: Configuration System (add distributed config)

## Notes
- DDP is simpler and works for most cases
- FSDP is needed only for very large models
- Gradient accumulation is cheaper than more GPUs
- Linear speedup is ideal but not always achieved
- Communication overhead increases with more GPUs
- torch.compile() can further speed up training
- For best performance, use NCCL backend on CUDA
