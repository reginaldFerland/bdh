# Task 0006: Create Inference Script with Generation Utilities

## Priority
**High** - Essential for making trained models usable for text generation

## Purpose
Build a standalone inference script and Python API for interactive text generation with trained BDH models. This provides an easy way to test models, experiment with generation parameters, and use models programmatically.

## Current State
- Generation only happens at the very end of training script
- Single hardcoded prompt: "To be or "
- Limited generation parameters (only top_k)
- No interactive mode
- No batch generation support
- No streaming output
- Generation method is basic (lines 158-172 in bdh.py)

## Expected Outcome
After implementing this task, the project should have:
1. Standalone `inference.py` script for interactive text generation
2. Support for multiple sampling strategies: temperature, top-k, top-p, beam search
3. Interactive CLI mode for conversational generation
4. Batch generation from multiple prompts
5. Streaming output for long generations
6. Easy-to-use Python API for programmatic use
7. Support for loading models from saved directories

## Detailed Requirements

### 1. Enhanced Generation Methods

Extend BDH class with advanced sampling:
```python
class BDH(nn.Module):
    # ... existing code ...
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        stream: bool = False,
    ):
        """Advanced text generation with multiple sampling strategies."""
        pass
    
    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate text from string prompt."""
        pass
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        **kwargs
    ) -> list[str]:
        """Generate text from multiple prompts."""
        pass
```

### 2. Sampling Strategies

#### Temperature Sampling
Scale logits by temperature before sampling:
- temperature < 1.0: More conservative, peaked distribution
- temperature = 1.0: Original distribution
- temperature > 1.0: More random, flatter distribution

#### Top-K Sampling
Keep only top K most probable tokens:
- Prevents sampling very unlikely tokens
- Typical values: 40-50

#### Top-P (Nucleus) Sampling
Sample from smallest set of tokens with cumulative probability >= P:
- More dynamic than top-k
- Typical values: 0.9-0.95

#### Repetition Penalty
Penalize tokens that have already been generated:
- Reduces repetitive text
- Typical values: 1.0-1.2

#### Beam Search
Maintain multiple hypotheses, select most probable sequence:
- More deterministic
- Better for tasks requiring coherence
- Typical beam size: 4-8

### 3. Interactive CLI Interface

```bash
$ python inference.py --model models/bdh-shakespeare --interactive

BDH Interactive Generation
Model: models/bdh-shakespeare (15M parameters)
Type 'quit' to exit, 'help' for commands

> Once upon a time
[Generated text...]

> /temp 0.8
Temperature set to 0.8

> /top_k 50
Top-k set to 50

> /help
Commands:
  /temp <value>     Set temperature
  /top_k <value>    Set top-k
  /top_p <value>    Set top-p
  /max_tokens <n>   Set max new tokens
  /reset            Reset conversation
  /quit             Exit
```

### 4. Batch Generation Mode

```bash
$ python inference.py \
    --model models/bdh-shakespeare \
    --prompts prompts.txt \
    --output generations.txt \
    --num_return_sequences 3 \
    --temperature 0.9
```

### 5. Streaming Generation

For long generations, show tokens as they're generated:
```python
for token in model.generate_stream(prompt, max_new_tokens=500):
    print(token, end='', flush=True)
```

## Implementation Steps

### Step 1: Enhance generate() Method

Update `bdh.py`:
```python
@torch.no_grad()
def generate(
    self,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    eos_token_id: int = None,
    pad_token_id: int = None,
) -> torch.Tensor:
    """Generate tokens with advanced sampling."""
    
    for _ in range(max_new_tokens):
        # Get logits
        logits, _ = self(idx)
        logits = logits[:, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                logits[0, token_id] /= repetition_penalty
        
        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        if do_sample:
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Check for EOS
        if eos_token_id is not None and idx_next.item() == eos_token_id:
            break
    
    return idx
```

### Step 2: Add Text Generation Wrappers

```python
@torch.no_grad()
def generate_text(
    self,
    prompt: str,
    max_new_tokens: int = 100,
    **kwargs
) -> str:
    """Generate text from string prompt."""
    # Encode prompt
    if self.tokenizer is None:
        input_ids = torch.tensor(
            bytearray(prompt, "utf-8"),
            dtype=torch.long,
            device=next(self.parameters()).device
        ).unsqueeze(0)
    else:
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=next(self.parameters()).device
        ).unsqueeze(0)
    
    # Generate
    output_ids = self.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)
    
    # Decode
    if self.tokenizer is None:
        return bytes(output_ids[0].to(torch.uint8).cpu()).decode(errors='backslashreplace')
    else:
        return self.tokenizer.decode(output_ids[0].tolist())

@torch.no_grad()
def generate_batch(
    self,
    prompts: list[str],
    max_new_tokens: int = 100,
    **kwargs
) -> list[str]:
    """Generate text from multiple prompts."""
    return [self.generate_text(prompt, max_new_tokens, **kwargs) for prompt in prompts]
```

### Step 3: Implement Streaming Generation

```python
@torch.no_grad()
def generate_stream(
    self,
    prompt: str,
    max_new_tokens: int = 100,
    **kwargs
):
    """Generate text with streaming output."""
    # Encode prompt
    if self.tokenizer is None:
        input_ids = torch.tensor(
            bytearray(prompt, "utf-8"),
            dtype=torch.long,
            device=next(self.parameters()).device
        ).unsqueeze(0)
    else:
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=next(self.parameters()).device
        ).unsqueeze(0)
    
    # Yield prompt
    yield prompt
    
    # Generate token by token
    for _ in range(max_new_tokens):
        logits, _ = self(input_ids)
        logits = logits[:, -1, :] / kwargs.get('temperature', 1.0)
        
        # Apply top-k
        if kwargs.get('top_k') is not None:
            v, _ = torch.topk(logits, min(kwargs['top_k'], logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
        # Decode and yield token
        if self.tokenizer is None:
            token_text = bytes([next_token.item()]).decode(errors='ignore')
        else:
            token_text = self.tokenizer.decode([next_token.item()])
        
        yield token_text
```

### Step 4: Create inference.py CLI Script

```python
#!/usr/bin/env python3
"""
BDH Inference Script - Interactive text generation with trained models.
"""
import argparse
from pathlib import Path
import sys

import torch
from bdh import BDH


def parse_args():
    parser = argparse.ArgumentParser(description="BDH Text Generation")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate from")
    parser.add_argument("--prompts", type=str, help="File with multiple prompts")
    parser.add_argument("--output", type=str, help="Output file for generations")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def interactive_mode(model, args):
    """Interactive generation mode."""
    print(f"\nBDH Interactive Generation")
    print(f"Model: {args.model}")
    print(f"Type 'quit' to exit, 'help' for commands\n")
    
    settings = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
    }
    
    while True:
        try:
            prompt = input("> ").strip()
            
            if not prompt:
                continue
            elif prompt == "quit":
                break
            elif prompt == "help":
                print_help()
                continue
            elif prompt.startswith("/"):
                handle_command(prompt, settings)
                continue
            
            # Generate
            if args.stream:
                for token in model.generate_stream(prompt, **settings):
                    print(token, end='', flush=True)
                print("\n")
            else:
                generated = model.generate_text(prompt, **settings)
                print(generated)
                print()
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def handle_command(command, settings):
    """Handle CLI commands."""
    parts = command.split()
    cmd = parts[0]
    
    if cmd == "/temp" and len(parts) == 2:
        settings['temperature'] = float(parts[1])
        print(f"Temperature set to {settings['temperature']}")
    elif cmd == "/top_k" and len(parts) == 2:
        settings['top_k'] = int(parts[1])
        print(f"Top-k set to {settings['top_k']}")
    elif cmd == "/top_p" and len(parts) == 2:
        settings['top_p'] = float(parts[1])
        print(f"Top-p set to {settings['top_p']}")
    elif cmd == "/max_tokens" and len(parts) == 2:
        settings['max_new_tokens'] = int(parts[1])
        print(f"Max tokens set to {settings['max_new_tokens']}")
    else:
        print("Unknown command. Type 'help' for available commands.")


def print_help():
    """Print help message."""
    print("""
Commands:
  /temp <value>      Set temperature (0.0-2.0)
  /top_k <value>     Set top-k (1-100)
  /top_p <value>     Set top-p (0.0-1.0)
  /max_tokens <n>    Set max new tokens
  quit               Exit
  help               Show this message
""")


def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = BDH.from_pretrained(args.model, device=args.device)
    model.eval()
    print("Model loaded successfully!")
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, args)
    
    # Single prompt
    elif args.prompt:
        if args.stream:
            for token in model.generate_stream(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ):
                print(token, end='', flush=True)
            print()
        else:
            generated = model.generate_text(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            print(generated)
    
    # Multiple prompts from file
    elif args.prompts:
        with open(args.prompts, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        generations = []
        for i, prompt in enumerate(prompts):
            print(f"Generating {i+1}/{len(prompts)}...")
            for _ in range(args.num_return_sequences):
                generated = model.generate_text(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                generations.append(f"Prompt: {prompt}\nGenerated: {generated}\n")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(generations))
            print(f"Saved to {args.output}")
        else:
            for gen in generations:
                print(gen)
    
    else:
        print("Please specify --interactive, --prompt, or --prompts")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 5: Create Python API Examples

Create `examples/generation_api.py`:
```python
"""Examples of using BDH generation API programmatically."""
from bdh import BDH

# Load model
model = BDH.from_pretrained("models/bdh-shakespeare")

# Simple generation
text = model.generate_text("To be or not to be", max_new_tokens=50)
print(text)

# With custom parameters
text = model.generate_text(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.9,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.1,
)
print(text)

# Batch generation
prompts = ["Hello", "The meaning of life", "In the beginning"]
texts = model.generate_batch(prompts, max_new_tokens=50, temperature=0.8)
for prompt, text in zip(prompts, texts):
    print(f"{prompt} -> {text}")

# Streaming generation
print("Streaming:")
for token in model.generate_stream("The future of AI", max_new_tokens=100):
    print(token, end='', flush=True)
```

## Testing Plan

1. **Test Temperature Sampling**
   - Generate with temp=0.1, 0.8, 1.5
   - Verify low temp is conservative, high temp is random
   - Check outputs are diverse

2. **Test Top-K Sampling**
   - Generate with top_k=1 (greedy), 10, 50
   - Verify quality and diversity
   - Check edge case: top_k > vocab_size

3. **Test Top-P Sampling**
   - Generate with top_p=0.5, 0.9, 0.99
   - Compare with top-k
   - Verify cumulative probability logic

4. **Test Repetition Penalty**
   - Generate long text with penalty=1.0, 1.2, 1.5
   - Check for reduced repetition
   - Verify doesn't break coherence

5. **Test Interactive Mode**
   - Start interactive session
   - Test various commands
   - Verify settings persist
   - Test error handling

6. **Test Batch Generation**
   - Generate from multiple prompts
   - Verify all complete successfully
   - Check memory usage is reasonable

7. **Test Streaming**
   - Generate with streaming
   - Verify tokens appear immediately
   - Check no buffering delays

8. **Test Edge Cases**
   - Empty prompt
   - Very long prompt (> block_size)
   - max_new_tokens=0
   - Invalid temperature values

## Code Example

Example usage after implementation:

```bash
# Interactive mode
python inference.py --model models/bdh-shakespeare --interactive --stream

# Single generation
python inference.py \
    --model models/bdh-shakespeare \
    --prompt "To be or not to be" \
    --temperature 0.8 \
    --top_k 50 \
    --max_new_tokens 100

# Batch generation
python inference.py \
    --model models/bdh-shakespeare \
    --prompts prompts.txt \
    --output generations.txt \
    --num_return_sequences 3 \
    --temperature 0.9

# Python API
from bdh import BDH
model = BDH.from_pretrained("models/bdh-shakespeare")
text = model.generate_text("Once upon a time", max_new_tokens=100, temperature=0.8, top_k=50)
print(text)
```

## Copilot Implementation Prompt

```
Create a comprehensive inference system for BDH:

1. Update the generate() method in bdh.py (currently lines 158-172) to add:
   - top_p (nucleus sampling): Sort logits, compute cumulative probs, filter tokens beyond threshold
   - repetition_penalty: Divide logits of already-generated tokens by penalty value
   - eos_token_id support: Stop generation when EOS token is generated
   - Keep existing temperature and top_k functionality

2. Add new methods to BDH class:
   - generate_text(prompt: str, **kwargs) -> str: Encode prompt (byte-level or tokenizer), call generate(), decode output
   - generate_batch(prompts: list[str], **kwargs) -> list[str]: Call generate_text() for each prompt
   - generate_stream(prompt: str, **kwargs): Yield tokens one by one as they're generated

3. Create inference.py script with:
   - argparse for CLI arguments: --model, --interactive, --prompt, --prompts, --output, --max_new_tokens, --temperature, --top_k, --top_p, --stream, --device
   - main() function that loads model with BDH.from_pretrained()
   - interactive_mode() function for interactive CLI
   - Support for commands: /temp, /top_k, /top_p, /max_tokens, quit, help
   - Single prompt mode: generate and print once
   - Batch mode: read prompts from file, generate for each, save to output file
   - Streaming mode: print tokens as generated with flush=True

4. For top-p (nucleus) sampling:
   - Sort logits descending
   - Compute cumulative softmax probabilities
   - Find cutoff where cumsum > top_p
   - Set filtered logits to -inf
   - Sample from remaining tokens

5. For repetition penalty:
   - Track tokens in generated sequence
   - For each already-generated token, divide its logit by penalty
   - Higher penalty = less repetition

6. Handle encoding/decoding:
   - If tokenizer is None: use bytearray(text, "utf-8") for encoding, bytes().decode() for decoding
   - If tokenizer exists: use tokenizer.encode() and tokenizer.decode()
   - Get device from next(model.parameters()).device

7. Interactive mode features:
   - Print welcome message with model info
   - Loop: read input, check for commands, generate, print
   - Commands update settings dict
   - Handle KeyboardInterrupt for clean exit

Example top-p implementation:
```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > top_p
# Keep at least one token
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0
```

The current generate() method is simple (lines 158-172). Extend it while preserving existing functionality.
```

## Files to Modify/Create
- **Modify**: `bdh.py` - Enhance generate() and add new generation methods
- **Create**: `inference.py` - CLI inference script
- **Create**: `examples/generation_api.py` - Python API examples

## Dependencies
- No new dependencies required
- Uses existing torch, torch.nn.functional
- Task 0005 (Model Export) for loading models

## Success Criteria
- [ ] Top-p sampling works correctly
- [ ] Repetition penalty reduces repetition
- [ ] Interactive mode works smoothly
- [ ] Streaming generation displays tokens immediately
- [ ] Batch generation handles multiple prompts
- [ ] Python API is easy to use
- [ ] CLI has good UX with help and commands
- [ ] Generated text quality is good
- [ ] No memory leaks in long generations

## Related Tasks
- **Task 0005**: Model Export (prerequisite for loading models)
- **Task 0002**: Tokenizer Support (for non-byte-level generation)
- **Task 0007**: API Server (will use these generation methods)

## Notes
- Top-p is generally preferred over top-k for quality
- Repetition penalty should be used carefully (1.0-1.2 range)
- Streaming is crucial for good UX with long generations
- Consider adding beam search in future for better quality
- Temperature and top-p/top-k can be combined effectively
