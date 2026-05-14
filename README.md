# EEG MNIST Classifier

## Installation

1. Install `uv` package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies from lockfile:

```bash
uv sync
```

3. Activate virtual environment

```bash
source .venv/bin/activate
```

## Development

You can start jupyter notebook with uv, or just use VSCode

```bash
uv run --with jupyter jupyter notebook
```

Project uses Makefile for some development tasks:

- `make lint` - Lint using ruff
- `make format` - Format using ruff
- `make clean` - Remove cache files
