# Open-weight models: Qwen3

Connect pretrained model inference to the mini GPT built in the previous chapter.
Status: scaffolds only; experiments and results are pending.

## Learning sequence and roadmap

| Notebook | Main question | Roadmap coverage |
| --- | --- | --- |
| [01: Hub and checkpoint](notebooks/01_hub_and_checkpoint.ipynb) | What do we download? | Hub, model cards, revisions, cache, configs and weights |
| [02: Tokenizer and forward](notebooks/02_tokenizer_and_forward.ipynb) | How does text become logits? | Local models, tokenizers, tensor shapes and real architecture |
| [03: Text generation](notebooks/03_text_generation.ipynb) | How does a forward pass become text? | Local generation, sampling and an introduction to KV caching |
| [04: Base versus post-trained](notebooks/04_base_vs_posttrained.ipynb) | What changes after post-training? | Instruction following, chat templates and controlled comparison |

Every notebook has ordered sections, code placeholders, a completion checklist
and space for evidence and conclusions. Work incrementally, predicting behavior
before running each experiment. Each notebook should run from a fresh kernel.
Extract reusable helpers into a future src/ only when they are understood.

Qwen is the first real architecture explored here. Llama, Gemma and Mistral remain
future comparisons. Identify RoPE, RMSNorm, SwiGLU and GQA, but defer their
from-scratch implementations to a later unit. Fine-tuning is also future work.

## Models and experiment record

- Base: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base).
- Post-trained: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
- Exact commit revisions: pending selection in the experiments.
- Record versions, OS, device, dtype, seeds, prompts and generation settings.
- Read both model cards. Post-training can include more than instruction tuning;
  do not attribute every behavioral difference to a single training stage.

## Environment and storage

Use the repository uv environment and launch Jupyter with `uv run jupyter lab`.
When starting notebook 01, add compatible Transformers and huggingface_hub
dependencies through uv and record their resolved versions.

Download configuration and tokenizer files first, then weights. Keep downloaded
weights in the Hugging Face cache outside Git. Check available memory before
loading models and load the comparison models sequentially if necessary.
No dependencies were added and no models were downloaded for this scaffold.

## Conclusions

- Notebook 01: pending.
- Notebook 02: pending.
- Notebook 03: pending.
- Notebook 04: pending.

Completion means the local forward and generation experiments work, their
behavior is explained, and the model comparison is recorded with its limitations.
