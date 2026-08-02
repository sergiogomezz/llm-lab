# LLM Lab

Personal lab for learning how large language models work from the ground up.
The repository will grow gradually as each topic is studied and implemented.

## Current focus: pretraining

The first project is a small GPT-like language model built from scratch with
PyTorch. Concepts are explored in notebooks first and moved to Python modules
once the implementation is understood and reusable.

```text
pretraining/
└── mini_gpt/
    ├── notebooks/  # One learning topic per notebook
    └── src/        # Consolidated, reusable implementations

experiments/        # Independent PyTorch and LLM explorations
data/               # Local datasets used by the projects
```

Future areas such as open-weight models, fine-tuning and inference will be
added when they become part of the active learning path.

## Workflow

1. Explore and understand a concept in a notebook.
2. Check tensor shapes, behavior and results.
3. Move stable implementations into `src/`.
4. Record conclusions and experiments alongside the code.

## Setup

Install the locked environment and start JupyterLab from the repository root:

```bash
uv sync
uv run jupyter lab
```

## Stack

- Python 3.12
- PyTorch
- JupyterLab
- uv
