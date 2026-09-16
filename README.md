# LLM Lab

Personal lab for learning how large language models work from the ground up.
The repository will grow gradually as each topic is studied and implemented.

## Current focus: open-weight models

The first project covers building and training a small GPT-like language model
from scratch with PyTorch. The next chapter explores the Hub, checkpoints, local
inference and base versus post-trained behavior with Qwen3.
Start with the [Qwen3 chapter](02_open_weight_models/qwen3/README.md).
Concepts are explored in notebooks first and moved to Python modules once the
implementation is understood and reusable.

```text
01_pretraining/
└── mini_gpt/
    ├── notebooks/  # One learning topic per notebook
    └── src/        # Consolidated, reusable implementations

02_open_weight_models/
└── qwen3/
    ├── README.md   # Chapter plan and roadmap coverage
    └── notebooks/  # Hub, forward pass, generation and model comparison

00_experiments/     # Independent PyTorch and LLM explorations
data/               # Local datasets used by the projects
```

Fine-tuning and deeper inference topics will be added when they become part of
the active learning path.

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
