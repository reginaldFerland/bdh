# Task 0003: Implement Checkpoint Saving/Loading During Training

## Priority
**High** - Critical for long training runs; without this, training cannot be resumed after interruption

## Purpose
Add robust checkpoint management to enable resuming interrupted training runs. Long training runs (hours/days) can be interrupted by hardware failures, OOM errors, or manual stops. Checkpoints allow continuing from where training left off without losing progress.

## Current State
- No checkpoint saving in `train.py`
- Training runs from scratch every time
- If training is interrupted, all progress is lost
- No way to resume training from a specific step
- Model weights are only available after training completes

## Expected Outcome
After implementing this task, the project should have:
1. Automatic checkpoint saving at regular intervals during training
2. Ability to resume training from the latest checkpoint
3. Checkpoint files containing: model weights, optimizer state, training step, loss history, config, and random states
4. Configurable checkpoint directory and save frequency
5. Automatic cleanup of old checkpoints (keep last N)
6. Support for saving "best" checkpoint based on validation loss

## Detailed Requirements

### 1. Checkpoint Contents
Each checkpoint should save:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # For mixed precision
    'step': current_step,
    'config': config_dict,
    'loss_history': loss_history,
    'best_val_loss': best_val_loss,
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
}
```

### 2. Checkpoint File Naming
Use clear, sortable naming convention:
```
checkpoints/
    checkpoint-0001000.pt  # Step 1000
    checkpoint-0002000.pt  # Step 2000
    checkpoint-0003000.pt  # Step 3000
    checkpoint-latest.pt   # Symlink or copy to latest
    checkpoint-best.pt     # Best validation loss
```

### 3. Configuration Options
Add these settings to training:
```python
CHECKPOINT_DIR = "checkpoints"
SAVE_CHECKPOINT_FREQ = 1000  # Save every N steps
KEEP_LAST_N_CHECKPOINTS = 5  # Delete older checkpoints
SAVE_BEST_CHECKPOINT = True  # Keep best validation loss separately
```

### 4. Resume Functionality
When starting training:
1. Check for existing checkpoints in checkpoint directory
2. If `--resume` flag or `--checkpoint_path` provided, load checkpoint
3. Restore all states: model, optimizer, scaler, step counter, RNG
4. Continue training from restored step
5. Preserve loss history and logging continuity

### 5. Validation-Based Checkpointing
Implement validation evaluation:
- Run validation every N steps
- Compute validation loss
- Save checkpoint if validation loss improves
- Track best validation loss across training

## Implementation Steps

### Step 1: Create Checkpoint Manager Class
Create `checkpoint.py` with checkpoint utilities:
```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def save_checkpoint(self, state: dict, step: int, is_best: bool = False):
        """Save checkpoint and manage cleanup."""
        pass
    
    def load_checkpoint(self, checkpoint_path: str = None) -> dict:
        """Load checkpoint from path or find latest."""
        pass
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        pass
    
    def cleanup_old_checkpoints(self):
        """Remove checkpoints beyond keep_last_n."""
        pass
```

### Step 2: Add Validation Loop
Create validation function in `train.py`:
```python
@torch.no_grad()
def evaluate(model, data_loader, eval_iters=100):
    """Evaluate model on validation set."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = data_loader.get_batch('val', BATCH_SIZE)
        with ctx:
            logits, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)
```

### Step 3: Update Training Loop
Modify `train.py` to:
- Initialize CheckpointManager
- Check for existing checkpoints on startup
- Save checkpoints at regular intervals
- Run validation periodically
- Track best validation loss
- Save best checkpoint separately

### Step 4: Add Command-Line Arguments
Add argparse arguments:
```python
parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
parser.add_argument('--checkpoint_path', type=str, help='Specific checkpoint to resume from')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--save_freq', type=int, default=1000)
parser.add_argument('--keep_last_n', type=int, default=5)
parser.add_argument('--eval_freq', type=int, default=500)
parser.add_argument('--eval_iters', type=int, default=100)
```

### Step 5: Implement Resume Logic
At training start:
```python
start_step = 0
best_val_loss = float('inf')

if args.resume or args.checkpoint_path:
    checkpoint = checkpoint_manager.load_checkpoint(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_step = checkpoint['step']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint['cuda_rng_state'] is not None:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    print(f"Resumed from step {start_step}")
```

### Step 6: Add Checkpoint Saving in Training Loop
```python
for step in range(start_step, MAX_ITERS):
    # ... training code ...
    
    # Evaluate and save checkpoints
    if step % args.eval_freq == 0 and step > 0:
        val_loss = evaluate(model, data_loader, args.eval_iters)
        print(f"Step {step}: val_loss = {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if step % args.save_freq == 0:
            checkpoint_manager.save_checkpoint(
                state={
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'step': step,
                    'config': vars(args),
                    'best_val_loss': best_val_loss,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                },
                step=step,
                is_best=is_best
            )
```

### Step 7: Implement Checkpoint Cleanup
After saving, remove old checkpoints:
```python
def cleanup_old_checkpoints(self):
    checkpoints = sorted(
        [f for f in self.checkpoint_dir.glob("checkpoint-*.pt") 
         if not f.name.endswith(('latest.pt', 'best.pt'))],
        key=lambda x: x.stat().st_mtime
    )
    while len(checkpoints) > self.keep_last_n:
        oldest = checkpoints.pop(0)
        oldest.unlink()
        print(f"Removed old checkpoint: {oldest.name}")
```

### Step 8: Add Checkpoint Averaging (Optional)
Implement checkpoint averaging for potentially better final model:
```python
def average_checkpoints(checkpoint_paths: list[str], output_path: str):
    """Average multiple checkpoint weights for better generalization."""
    checkpoints = [torch.load(path) for path in checkpoint_paths]
    averaged_state = {}
    
    for key in checkpoints[0]['model_state_dict'].keys():
        averaged_state[key] = torch.stack(
            [ckpt['model_state_dict'][key].float() for ckpt in checkpoints]
        ).mean(dim=0)
    
    torch.save({'model_state_dict': averaged_state}, output_path)
```

## Testing Plan

1. **Test Basic Checkpoint Saving**
   - Train for 500 steps
   - Verify checkpoint file is created
   - Check checkpoint contains all required keys
   - Verify file size is reasonable

2. **Test Resume Functionality**
   - Train for 1000 steps, save checkpoint
   - Stop training
   - Resume with `--resume` flag
   - Verify training continues from step 1000
   - Check loss continuity

3. **Test Random State Restoration**
   - Train with seed, save checkpoint
   - Resume from checkpoint
   - Verify same random batches are generated
   - Compare with non-resumed training (should differ)

4. **Test Checkpoint Cleanup**
   - Train for 10,000 steps with save_freq=1000, keep_last_n=3
   - Verify only 3 checkpoints remain (plus latest/best)
   - Check oldest checkpoints are deleted

5. **Test Best Checkpoint**
   - Train with validation
   - Manually check that checkpoint-best.pt has lowest validation loss
   - Verify best checkpoint is preserved during cleanup

6. **Test Edge Cases**
   - Resume from corrupted checkpoint (handle gracefully)
   - Resume with different config (warn user)
   - Disk full during save (handle error)
   - Missing checkpoint directory (create automatically)

## Code Example

Example usage after implementation:

```bash
# Start fresh training with checkpointing
python train.py \
    --checkpoint_dir checkpoints/run1 \
    --save_freq 1000 \
    --eval_freq 500 \
    --keep_last_n 5

# Resume from latest checkpoint
python train.py --resume --checkpoint_dir checkpoints/run1

# Resume from specific checkpoint
python train.py --checkpoint_path checkpoints/run1/checkpoint-005000.pt

# Average last 3 checkpoints for final model
python average_checkpoints.py \
    --checkpoints checkpoints/run1/checkpoint-008000.pt \
                  checkpoints/run1/checkpoint-009000.pt \
                  checkpoints/run1/checkpoint-010000.pt \
    --output models/bdh-averaged.pt
```

## Copilot Implementation Prompt

```
Implement checkpoint saving and loading for the BDH training script:

1. Create a CheckpointManager class in checkpoint.py that:
   - Saves checkpoints to a specified directory with naming pattern "checkpoint-{step:07d}.pt"
   - Saves special checkpoints: "checkpoint-latest.pt" and "checkpoint-best.pt"
   - Implements cleanup to keep only last N checkpoints
   - Has methods: save_checkpoint(state, step, is_best), load_checkpoint(path), get_latest_checkpoint()
   - Creates checkpoint directory if it doesn't exist

2. Each checkpoint should include:
   - model.state_dict()
   - optimizer.state_dict()
   - scaler.state_dict() (for mixed precision training)
   - current training step
   - config/hyperparameters
   - best validation loss
   - torch.get_rng_state() and torch.cuda.get_rng_state()

3. Update train.py to:
   - Add command-line arguments: --resume, --checkpoint_path, --checkpoint_dir, --save_freq, --keep_last_n, --eval_freq, --eval_iters
   - Initialize CheckpointManager at start
   - Check for existing checkpoints and load if --resume is specified
   - Restore all state from checkpoint: model, optimizer, scaler, step, RNG states
   - Save checkpoints every save_freq steps during training loop (currently lines 102-126)
   - Run validation every eval_freq steps
   - Track best validation loss and save best checkpoint separately
   - Clean up old checkpoints automatically

4. Add evaluation function:
   - Create evaluate() function that computes validation loss
   - Use torch.no_grad() for efficiency
   - Run for eval_iters iterations
   - Return average validation loss
   - Set model back to training mode after eval

5. Update training loop to:
   - Start from start_step (0 or loaded from checkpoint)
   - Save checkpoint at regular intervals
   - Run evaluation periodically
   - Track and print validation loss
   - Update best checkpoint when validation improves

6. Handle edge cases:
   - Checkpoint file doesn't exist (start fresh)
   - Corrupted checkpoint (print error, start fresh)
   - Missing checkpoint directory (create it)
   - Disk full during save (catch exception, warn user)

Current training loop is in train.py lines 95-126. The model is compiled with torch.compile() on line 92. Make sure checkpoint saving is compatible with compiled models by using model.state_dict() which automatically handles this.

Example checkpoint structure:
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'step': step,
    'config': {...},
    'best_val_loss': 2.5,
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state(),
}
```
```

## Files to Modify/Create
- **Create**: `checkpoint.py` - Checkpoint management utilities
- **Modify**: `train.py` - Add checkpointing to training loop
- **Create**: `average_checkpoints.py` - Optional utility for checkpoint averaging

## Dependencies
- `torch` (already present)
- `pathlib` (Python standard library)
- No new external dependencies needed

## Success Criteria
- [ ] Checkpoints are saved automatically during training
- [ ] Training can be resumed from latest checkpoint
- [ ] Random state is correctly restored (deterministic resume)
- [ ] Old checkpoints are cleaned up automatically
- [ ] Best checkpoint is saved and preserved
- [ ] Validation loss is computed and logged
- [ ] Checkpoint loading handles missing files gracefully
- [ ] Works with torch.compile() compiled models
- [ ] Documentation includes resume examples

## Related Tasks
- **Task 0004**: Training Monitoring (will log checkpoint events)
- **Task 0005**: Model Export (will use checkpoints as source)
- **Task 0009**: Configuration System (will integrate checkpoint config)

## Notes
- Checkpoints can be large (hundreds of MB to GBs for large models)
- Consider adding compression for checkpoint files
- For very frequent saving, consider async saving to avoid blocking training
- Checkpoint averaging typically uses last 3-10 checkpoints
- Best checkpoint is based on validation loss, not training loss
