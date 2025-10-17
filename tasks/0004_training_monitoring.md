# Task 0004: Add Training Monitoring and Logging

## Priority
**Medium** - Important for tracking experiments and debugging training issues

## Purpose
Integrate comprehensive experiment tracking and monitoring to visualize training progress, compare experiments, and debug issues. This enables systematic hyperparameter tuning and provides insights into model behavior during training.

## Current State
- Only basic print statements for loss logging
- Loss is averaged over LOG_FREQ steps and printed
- No visualization of training progress
- No experiment tracking or comparison
- No metric history saved
- No way to monitor gradient norms, learning rates, or other diagnostics
- Sample generation only happens at the very end

## Expected Outcome
After implementing this task, the project should have:
1. Integration with Weights & Biases (W&B) or TensorBoard for experiment tracking
2. Comprehensive metric logging: train loss, val loss, learning rate, gradient norms
3. Periodic evaluation on validation set with perplexity calculation
4. Sample text generation during training to monitor quality
5. Configurable logging frequency and verbosity
6. Ability to compare multiple training runs
7. Visual dashboards for real-time monitoring

## Detailed Requirements

### 1. Metrics to Log

#### Training Metrics (Every Step)
- Training loss
- Learning rate
- Gradient norm (global and per-parameter group)
- Step time / throughput (tokens/sec)

#### Validation Metrics (Periodic)
- Validation loss
- Validation perplexity
- Token accuracy (top-1)

#### Model Metrics (Periodic)
- Parameter norms
- Activation statistics
- Weight histogram

#### Generation Samples (Periodic)
- Generated text samples from fixed prompts
- Generation quality assessment

### 2. Logging Backends

Support both W&B and TensorBoard:
```python
class Logger:
    def __init__(self, backend: str = "wandb", project_name: str = "bdh-training"):
        """Initialize logging backend (wandb or tensorboard)."""
        pass
    
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics at current step."""
        pass
    
    def log_text(self, name: str, text: str, step: int):
        """Log generated text samples."""
        pass
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log weight/activation histograms."""
        pass
    
    def finish(self):
        """Clean up and finish logging."""
        pass
```

### 3. Evaluation Loop

Implement comprehensive evaluation:
```python
@torch.no_grad()
def evaluate(model, data_loader, eval_iters=100):
    """Run validation evaluation."""
    model.eval()
    losses = []
    accuracies = []
    
    for _ in range(eval_iters):
        x, y = data_loader.get_batch('val', batch_size)
        with ctx:
            logits, loss = model(x, y)
        
        losses.append(loss.item())
        
        # Calculate token accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y).float().mean()
        accuracies.append(accuracy.item())
    
    model.train()
    
    return {
        'val_loss': np.mean(losses),
        'val_perplexity': np.exp(np.mean(losses)),
        'val_accuracy': np.mean(accuracies),
    }
```

### 4. Sample Generation During Training

Generate samples periodically to monitor quality:
```python
@torch.no_grad()
def generate_samples(model, prompts: list[str], max_tokens: int = 100):
    """Generate text from multiple prompts."""
    model.eval()
    samples = {}
    
    for prompt in prompts:
        # Encode prompt
        input_ids = encode_prompt(prompt)
        # Generate
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens)
        # Decode
        generated_text = decode_output(output_ids)
        samples[prompt] = generated_text
    
    model.train()
    return samples
```

### 5. Configuration Options

Add logging configuration:
```python
# Logging settings
LOG_BACKEND = "wandb"  # "wandb", "tensorboard", or "none"
WANDB_PROJECT = "bdh-training"
WANDB_ENTITY = None  # Your W&B username/team
LOG_TRAIN_FREQ = 10  # Log training metrics every N steps
EVAL_FREQ = 500  # Run validation every N steps
EVAL_ITERS = 100  # Number of validation batches
GENERATE_FREQ = 1000  # Generate samples every N steps
GENERATION_PROMPTS = [
    "To be or not to be",
    "Once upon a time",
    "In a galaxy far, far away",
]
```

## Implementation Steps

### Step 1: Add Dependencies
Update `requirements.txt`:
```
wandb>=0.16.0
tensorboard>=2.15.0  # Optional alternative
```

### Step 2: Create Logger Class
Create `logger.py` with unified logging interface:
```python
class Logger:
    """Unified interface for experiment logging."""
    
    def __init__(self, backend: str, config: dict, **kwargs):
        self.backend = backend
        if backend == "wandb":
            import wandb
            wandb.init(project=kwargs.get('project'), config=config)
            self.log_fn = wandb.log
        elif backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(kwargs.get('log_dir'))
        elif backend == "none":
            self.log_fn = lambda *args, **kwargs: None
    
    def log_metrics(self, metrics: dict, step: int):
        # Implementation for each backend
        pass
```

### Step 3: Implement Evaluation Function
Add to `train.py`:
```python
@torch.no_grad()
def evaluate(model, data_loader, eval_iters=100):
    """Comprehensive evaluation on validation set."""
    model.eval()
    metrics = {
        'losses': [],
        'accuracies': [],
    }
    
    for _ in range(eval_iters):
        x, y = data_loader.get_batch('val', BATCH_SIZE)
        with ctx:
            logits, loss = model(x, y)
        
        metrics['losses'].append(loss.item())
        
        # Token accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y).float().mean()
        metrics['accuracies'].append(accuracy.item())
    
    model.train()
    
    avg_loss = np.mean(metrics['losses'])
    return {
        'val/loss': avg_loss,
        'val/perplexity': np.exp(avg_loss),
        'val/accuracy': np.mean(metrics['accuracies']),
    }
```

### Step 4: Add Gradient Monitoring
Implement gradient tracking:
```python
def get_grad_norm(model):
    """Calculate global gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
```

### Step 5: Implement Sample Generation
Add generation utilities:
```python
@torch.no_grad()
def generate_samples(model, prompts, tokenizer, max_tokens=100, **gen_kwargs):
    """Generate text samples from prompts."""
    model.eval()
    samples = {}
    
    for prompt in prompts:
        # Encode prompt (byte-level or tokenizer)
        if tokenizer is None:
            input_ids = torch.tensor(
                bytearray(prompt, "utf-8"), 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0)
        else:
            input_ids = tokenizer.encode(prompt)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens, **gen_kwargs)
        
        # Decode
        if tokenizer is None:
            generated_text = bytes(output_ids.to(torch.uint8).cpu().squeeze(0)).decode(errors='backslashreplace')
        else:
            generated_text = tokenizer.decode(output_ids.squeeze(0).tolist())
        
        samples[f"sample/{prompt[:30]}"] = generated_text
    
    model.train()
    return samples
```

### Step 6: Update Training Loop
Integrate logging into training:
```python
# Initialize logger
logger = Logger(
    backend=LOG_BACKEND,
    config={
        'n_layer': BDH_CONFIG.n_layer,
        'n_embd': BDH_CONFIG.n_embd,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'max_iters': MAX_ITERS,
    },
    project=WANDB_PROJECT,
)

# Training loop
for step in range(start_step, MAX_ITERS):
    # Forward pass
    with ctx:
        logits, loss = model(x, y)
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Log training metrics
    if step % LOG_TRAIN_FREQ == 0:
        grad_norm = get_grad_norm(model)
        logger.log_metrics({
            'train/loss': loss.item(),
            'train/lr': optimizer.param_groups[0]['lr'],
            'train/grad_norm': grad_norm,
            'train/step': step,
        }, step=step)
    
    # Evaluation
    if step % EVAL_FREQ == 0 and step > 0:
        eval_metrics = evaluate(model, data_loader, EVAL_ITERS)
        logger.log_metrics(eval_metrics, step=step)
        print(f"Step {step}: val_loss={eval_metrics['val/loss']:.4f}, "
              f"val_ppl={eval_metrics['val/perplexity']:.2f}")
    
    # Generate samples
    if step % GENERATE_FREQ == 0 and step > 0:
        samples = generate_samples(model, GENERATION_PROMPTS, tokenizer=None)
        for name, text in samples.items():
            logger.log_text(name, text, step=step)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    # Get next batch
    x, y = get_batch("train")

logger.finish()
```

### Step 7: Add Command-Line Arguments
Add to argparse:
```python
parser.add_argument('--log_backend', type=str, default='wandb', 
                   choices=['wandb', 'tensorboard', 'none'])
parser.add_argument('--wandb_project', type=str, default='bdh-training')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--log_train_freq', type=int, default=10)
parser.add_argument('--eval_freq', type=int, default=500)
parser.add_argument('--eval_iters', type=int, default=100)
parser.add_argument('--generate_freq', type=int, default=1000)
```

### Step 8: Add W&B Artifacts (Optional)
For model checkpoints and dataset versioning:
```python
# Save checkpoint as artifact
if logger.backend == "wandb":
    artifact = wandb.Artifact('model-checkpoint', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
```

## Testing Plan

1. **Test W&B Integration**
   - Train with `--log_backend wandb`
   - Verify metrics appear in W&B dashboard
   - Check text samples are logged
   - Verify config is saved

2. **Test TensorBoard Integration**
   - Train with `--log_backend tensorboard`
   - Open tensorboard and verify metrics
   - Check histograms appear
   - Verify scalars are plotted

3. **Test Evaluation Loop**
   - Run evaluation manually
   - Verify perplexity calculation is correct
   - Check accuracy computation
   - Ensure no memory leaks

4. **Test Sample Generation**
   - Generate samples during training
   - Verify text quality improves over time
   - Check same prompts produce different outputs
   - Monitor for repetition or degeneration

5. **Test Gradient Monitoring**
   - Log gradient norms
   - Check for exploding/vanishing gradients
   - Verify gradient clipping works if added

6. **Test Performance Impact**
   - Measure training speed with/without logging
   - Ensure logging doesn't significantly slow training
   - Check memory usage is reasonable

## Code Example

Example usage after implementation:

```bash
# Train with W&B logging
python train.py \
    --log_backend wandb \
    --wandb_project bdh-experiments \
    --log_train_freq 10 \
    --eval_freq 500 \
    --generate_freq 1000

# Train with TensorBoard
python train.py \
    --log_backend tensorboard \
    --log_train_freq 10 \
    --eval_freq 500

# Train without logging (fast)
python train.py --log_backend none

# View TensorBoard
tensorboard --logdir runs/
```

## Copilot Implementation Prompt

```
Implement comprehensive training monitoring and logging for the BDH model:

1. Create a Logger class in logger.py that:
   - Supports both Weights & Biases (wandb) and TensorBoard backends
   - Has methods: log_metrics(metrics_dict, step), log_text(name, text, step), log_histogram(name, tensor, step), finish()
   - Initializes based on backend parameter ("wandb", "tensorboard", or "none")
   - For wandb: use wandb.init() with project name and config
   - For tensorboard: use SummaryWriter from torch.utils.tensorboard
   - Handles backend-specific formatting automatically

2. Add evaluation function to train.py:
   - @torch.no_grad() decorated function evaluate(model, data_loader, eval_iters)
   - Computes validation loss, perplexity (exp(loss)), and token accuracy
   - Returns dict with keys: 'val/loss', 'val/perplexity', 'val/accuracy'
   - Sets model.eval() and restores model.train() after
   - Uses the same ctx and device as training

3. Add gradient monitoring function:
   - get_grad_norm(model) that computes global gradient L2 norm
   - Iterate over model.parameters() and aggregate grad norms
   - Return float value

4. Add sample generation function:
   - generate_samples(model, prompts, tokenizer, max_tokens, **kwargs)
   - Takes list of prompt strings
   - Uses byte-level encoding if tokenizer is None (current behavior)
   - Calls model.generate() with specified kwargs
   - Decodes output back to text
   - Returns dict mapping prompt to generated text

5. Update training loop in train.py (currently lines 102-126):
   - Initialize Logger at start with config dict containing all hyperparameters
   - Log training metrics every LOG_TRAIN_FREQ steps: loss, learning rate, gradient norm
   - Run evaluate() every EVAL_FREQ steps and log results
   - Generate samples every GENERATE_FREQ steps and log as text
   - Call logger.finish() at end of training

6. Add command-line arguments:
   - --log_backend (wandb/tensorboard/none)
   - --wandb_project and --wandb_entity
   - --log_train_freq, --eval_freq, --eval_iters, --generate_freq
   - Default prompts: ["To be or not to be", "Once upon a time"]

7. Ensure minimal performance impact:
   - Only compute expensive metrics (grad norm, eval) at specified frequencies
   - Use torch.no_grad() for evaluation
   - Don't block training for logging

The current training loop prints loss every LOG_FREQ=100 steps. Extend this with structured logging while maintaining the same print statements for backward compatibility.

Current generation code is at the end of train.py (lines 120-126). Adapt this for periodic sampling during training.
```

## Files to Modify/Create
- **Create**: `logger.py` - Unified logging interface
- **Modify**: `train.py` - Add evaluation, monitoring, and logging
- **Modify**: `requirements.txt` - Add wandb and tensorboard

## Dependencies
- `wandb>=0.16.0` (for W&B backend)
- `tensorboard>=2.15.0` (for TensorBoard backend)
- Task 0001 (Datasets) recommended for validation split
- Task 0003 (Checkpoints) integrates well with logging

## Success Criteria
- [ ] W&B integration works and logs metrics
- [ ] TensorBoard integration works and logs metrics
- [ ] Validation loss and perplexity are computed correctly
- [ ] Sample generation runs periodically during training
- [ ] Gradient norms are tracked
- [ ] Multiple experiments can be compared in dashboard
- [ ] Logging doesn't significantly slow training (<5% overhead)
- [ ] Generated samples show improvement over training
- [ ] Documentation includes W&B/TensorBoard examples

## Related Tasks
- **Task 0003**: Checkpoint Management (log checkpoint saves)
- **Task 0001**: HuggingFace Datasets (needed for validation split)
- **Task 0009**: Configuration System (will integrate logging config)

## Notes
- W&B is recommended for cloud-based experiment tracking
- TensorBoard is better for offline/local development
- Consider adding learning rate scheduling and logging LR curves
- Sample generation can be expensive; adjust frequency based on model size
- For large models, consider logging only subset of parameters/gradients
