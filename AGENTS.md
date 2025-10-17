# Repository Guidelines

## Project Structure & Module Organization
Core model code lives in `bdh.py`, which defines `BDHConfig`, the attention module, and the `BDH` network. Training entry points are in `train.py`, which downloads data, builds batches, and runs the optimizer; add new scripts alongside it or under a new `scripts/` directory. Configuration references live in `docs/settings.md`; update those tables whenever defaults move. Visual assets are kept in `figs/`, and roadmap work is tracked in `tasks/` plus `TODO.md`—cite the relevant item when contributing. Keep generated corpora (e.g., `input.txt`) out of Git by placing new data under `data/` with ignore rules.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment; Windows uses `.venv\Scripts\activate`.
- `pip install -r requirements.txt`: install PyTorch, NumPy, and Requests.
- `python train.py`: download tiny Shakespeare if missing, compile the model, and execute the default training loop.
- `CUDA_VISIBLE_DEVICES=0 python train.py`: run on a specific GPU when several are present.
- `python train.py | tee runs/latest.log`: capture loss curves for later review during manual tests.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, snake_case functions, and UpperCamelCase classes. Reuse dataclasses for configuration and add type hints to public entry points. Group imports as standard library, third-party, then local modules. Document tensor shapes with brief comments and keep parameter initialization near their consumers, mirroring `BDH.__init__`.

## Testing Guidelines
There is no dedicated test suite yet; create one under `tests/` using `pytest` as you add modules. Until automation lands, rely on abbreviated training runs to confirm loss trends and regression behaviour. Fix random seeds in tests that touch stochastic code and document acceptable numeric tolerances.

## Commit & Pull Request Guidelines
Recent history favors short, imperative, Title Case subjects such as “Add Checkpoint Management.” Keep the first line under 72 characters and elaborate in the body if necessary. In pull requests, link the relevant `tasks/` or `TODO.md` item, attach key evidence (loss curves, sample outputs), and note required environment tweaks like GPU memory or alternate datasets. Include screenshots or logs only when they materially help reviewers.

## Configuration & Data Tips
Treat `docs/settings.md` as the canonical source before touching hyperparameters; refresh its quick-reference tables when defaults change. External data downloads rely on `requests`, so keep endpoints configurable for offline mirrors. When exporting checkpoints, place them in an ignored `artifacts/` directory and record naming patterns for reproducibility.
