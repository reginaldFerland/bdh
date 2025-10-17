# Task 0009: Create Comprehensive Configuration System

## Priority
**High** - Makes all other features easier to use and maintain

## Purpose
Replace hardcoded hyperparameters and command-line arguments with a flexible, hierarchical configuration system using YAML files. This enables easy experiment management, reproducibility, and sharing of training configurations.

## Current State
- All hyperparameters hardcoded in train.py
- Growing number of command-line arguments
- No standard way to save/share configurations
- Difficult to reproduce experiments
- No configuration validation

## Expected Outcome
1. YAML-based configuration files for all settings
2. Configuration dataclasses with validation
3. CLI arguments override config file values
4. Example configs for common scenarios
5. Config saving with checkpoints
6. Configuration inheritance and composition

## Implementation Steps

### Step 1: Create Configuration Dataclasses

Create `config.py`:
```python
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.1
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    tokenizer_type: str = "byte"
    tokenizer_vocab_size: int = 256


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    max_iters: int = 30000
    batch_size: int = 8
    block_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 1000
    lr_decay_iters: int = 30000
    min_lr: float = 1e-4


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset_name: str = "shakespeare"
    dataset_config: Optional[str] = None
    text_column: str = "text"
    streaming: bool = False
    train_split: float = 0.9
    num_workers: int = 4


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 1000
    keep_last_n: int = 5
    eval_freq: int = 500
    eval_iters: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_backend: str = "wandb"
    wandb_project: str = "bdh-training"
    wandb_entity: Optional[str] = None
    log_train_freq: int = 10
    generate_freq: int = 1000
    generation_prompts: List[str] = field(default_factory=lambda: [
        "To be or not to be",
        "Once upon a time",
    ])


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    use_ddp: bool = False
    use_fsdp: bool = False
    gradient_accumulation_steps: int = 1


@dataclass
class BDHTrainingConfig:
    """Complete training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # System settings
    device: str = "auto"
    dtype: str = "float16"
    compile: bool = False
    seed: int = 1337
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BDHTrainingConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BDHTrainingConfig':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            checkpoint=CheckpointConfig(**config_dict.get('checkpoint', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            distributed=DistributedConfig(**config_dict.get('distributed', {})),
            device=config_dict.get('device', 'auto'),
            dtype=config_dict.get('dtype', 'float16'),
            compile=config_dict.get('compile', False),
            seed=config_dict.get('seed', 1337),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'checkpoint': asdict(self.checkpoint),
            'logging': asdict(self.logging),
            'distributed': asdict(self.distributed),
            'device': self.device,
            'dtype': self.dtype,
            'compile': self.compile,
            'seed': self.seed,
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
```

### Step 2: Create Example Configuration Files

Create `configs/default.yaml`:
```yaml
# Default BDH training configuration

model:
  n_layer: 6
  n_embd: 256
  n_head: 4
  dropout: 0.1
  vocab_size: 256
  tokenizer_type: "byte"

training:
  max_iters: 30000
  batch_size: 8
  block_size: 512
  learning_rate: 0.001
  weight_decay: 0.1
  grad_clip: 1.0
  warmup_iters: 1000

data:
  dataset_name: "shakespeare"
  streaming: false
  train_split: 0.9

checkpoint:
  checkpoint_dir: "checkpoints"
  save_freq: 1000
  keep_last_n: 5
  eval_freq: 500

logging:
  log_backend: "wandb"
  wandb_project: "bdh-training"
  log_train_freq: 10

distributed:
  use_ddp: false
  gradient_accumulation_steps: 1

device: "auto"
dtype: "float16"
compile: true
seed: 1337
```

Create `configs/large.yaml`:
```yaml
# Large model configuration

model:
  n_layer: 12
  n_embd: 768
  n_head: 12
  dropout: 0.1
  vocab_size: 32000
  tokenizer_type: "bpe"

training:
  max_iters: 100000
  batch_size: 4
  block_size: 1024
  learning_rate: 0.0003
  weight_decay: 0.1

data:
  dataset_name: "openwebtext"
  streaming: true

distributed:
  use_ddp: true
  gradient_accumulation_steps: 8

compile: true
```

### Step 3: Update train.py

```python
import argparse
from config import BDHTrainingConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    # Allow overriding any config value
    parser.add_argument('--model.n_layer', type=int, help='Override model layers')
    parser.add_argument('--training.learning_rate', type=float, help='Override LR')
    parser.add_argument('--data.dataset_name', type=str, help='Override dataset')
    # ... more overrides as needed
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config from file
    config = BDHTrainingConfig.from_yaml(args.config)
    
    # Override with CLI arguments
    for key, value in vars(args).items():
        if value is not None and '.' in key:
            section, param = key.split('.')
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, param):
                    setattr(section_config, param, value)
    
    # Save config with checkpoint
    config.save_yaml(f"{config.checkpoint.checkpoint_dir}/config.yaml")
    
    # Use config values
    model = BDH(BDHConfig(
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        # ... other params
    ))
    
    # Training with config values
    for step in range(config.training.max_iters):
        # Use config.training.batch_size, etc.
        pass
```

## Testing Plan

1. Load config from YAML
2. Override with CLI args
3. Save config with checkpoint
4. Validate config values
5. Test example configs work

## Copilot Implementation Prompt

```
Create a comprehensive configuration system for BDH:

1. Create config.py with dataclasses for:
   - ModelConfig: architecture parameters (n_layer, n_embd, etc.)
   - TrainingConfig: training hyperparameters (lr, batch_size, etc.)
   - DataConfig: dataset settings
   - CheckpointConfig: checkpointing settings
   - LoggingConfig: logging settings
   - DistributedConfig: distributed training settings
   - BDHTrainingConfig: top-level config combining all above

2. Add methods to BDHTrainingConfig:
   - from_yaml(path): Load from YAML file
   - from_dict(dict): Create from dictionary
   - to_dict(): Convert to dictionary
   - save_yaml(path): Save to YAML file

3. Create configs/ directory with example YAML files:
   - default.yaml: Small model, Shakespeare
   - large.yaml: Large model, distributed training
   - wikitext.yaml: Medium model, WikiText dataset

4. Update train.py:
   - Add --config argument to load config file
   - Support CLI overrides: --model.n_layer, --training.learning_rate, etc.
   - Replace hardcoded values with config.section.parameter
   - Save config alongside checkpoints

5. Use dataclasses with type hints and default values
6. Validate config values (positive integers, valid ranges)
7. Support nested configuration structure
8. Make configs composable (inherit from default)

All current hardcoded values in train.py should move to config system.
```

## Files to Create
- **Create**: `config.py`
- **Create**: `configs/default.yaml`, `configs/large.yaml`, `configs/wikitext.yaml`
- **Modify**: `train.py`

## Success Criteria
- [ ] Can load config from YAML
- [ ] CLI overrides work
- [ ] Config saved with checkpoints
- [ ] Example configs work
- [ ] No hardcoded hyperparameters remain

## Related Tasks
All tasks benefit from configuration system

## Notes
- Use dataclasses for type safety
- YAML is human-readable and editable
- Consider using Hydra for more advanced needs
