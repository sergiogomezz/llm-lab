# LLM Lab

Personal lab for learning how large language models work from the ground up.
The repository will grow gradually as each topic is studied and implemented.

## Current focus: open-weight models

The first project covers building and training a small GPT-like language model
from scratch with PyTorch. The next chapter explores the Hub, checkpoints, local
inference and base versus post-trained behavior with Qwen3.
Start with the [Qwen3 chapter](02_open_weight_models/qwen3/notebooks/01_hub_and_checkpoint.ipynb).
Concepts are explored in notebooks first and moved to Python modules once the
implementation is understood and reusable.

## Learning roadmap

The folders below map out the full learning path. Future stages contain topic
checklists; add experiments incrementally as each stage becomes the focus.

| Stage | Focus | Status |
| --- | --- | --- |
| [01: Pretraining](01_pretraining/README.md) | Build and train a GPT from scratch; PyTorch foundations | Core learning unit completed |
| [02: Open-weight models](02_open_weight_models/) | Hub, checkpoints, tokenizers, local inference and model comparisons | Current focus; Qwen3 checkpoint inspection completed |
| [03: Fine-tuning](03_fine_tuning/README.md) | Full fine-tuning, LoRA, QLoRA and instruction tuning | Planned |
| [04: Inference and deployment](04_inference_and_deployment/README.md) | KV caching, quantization, serving and performance | Planned |
| [05: LLM internals and optimization](05_llm_internals_and_optimization/README.md) | Modern components, attention, decoding and scheduling | Planned |
| [06: Alignment and evaluation](06_alignment_and_evaluation/README.md) | RLHF, preference optimization, evaluation and safety | Planned |

```text
01_pretraining/
└── mini_gpt/
    ├── notebooks/  # One learning topic per notebook
    └── src/        # Consolidated, reusable implementations

02_open_weight_models/
└── qwen3/
    └── notebooks/  # Hub, forward pass, generation and model comparison

03_fine_tuning/
04_inference_and_deployment/
05_llm_internals_and_optimization/
06_alignment_and_evaluation/

00_experiments/     # Independent PyTorch and LLM explorations
data/               # Local datasets used by the projects
```

Stage READMEs and notebook introductions describe their scope. Check off a topic only after its
implementation or experiment has been verified and its conclusions recorded.
The numbering follows the learning order, with open-weight models as a separate
stage between foundations and fine-tuning.

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
