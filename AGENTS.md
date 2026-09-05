# AGENTS.md

## Repository purpose

`llm-lab` is a personal learning laboratory for understanding large language
models from first principles through implementation and experimentation. The
main objective is not to assemble demos as quickly as possible, but to build a
reliable mental model of how each component works and how the components fit
together in real systems.

Favor learning value, clarity, and reproducible experiments over abstraction,
feature count, or production polish.

## Learning roadmap

Use this roadmap to understand the long-term direction of the repository. It is
not a requirement to work on every topic at once. Prefer the current focus and
introduce later topics only when they are relevant.

### 1. Foundations

- Finish *Build a Large Language Model (From Scratch)*.
- Implement a GPT-style model from scratch in PyTorch.
- Train a small GPT and run controlled experiments with it.
- Develop a solid practical understanding of PyTorch.

### 1.1 Open-weight models

- Learn the Hugging Face Hub and model checkpoint workflow.
- Run open-weight models locally.
- Understand model configurations, weights, and tokenizers.
- Explore real architectures such as Llama, Qwen, Gemma, and Mistral.
- Compare base and instruction-tuned models.

### 2. Fine-tuning

- Learn the Hugging Face ecosystem, especially Transformers and Datasets.
- Understand and perform full fine-tuning.
- Learn LoRA and QLoRA.
- Perform instruction tuning.

### 3. Inference and deployment

- Understand KV caching, vLLM, and PagedAttention.
- Learn GGUF and llama.cpp.
- Explore quantization techniques and their trade-offs.
- Serve models through APIs with batching and continuous batching.
- Measure and reason about latency, throughput, and memory usage.

### 4. LLM internals and optimization

- Study modern components such as RoPE, RMSNorm, and SwiGLU.
- Understand KV-cache behavior in detail.
- Study multi-query and grouped-query attention (MQA/GQA).
- Learn how FlashAttention works and why it improves performance.
- Explore speculative decoding and multi-token prediction (MTP).
- Study mixture-of-experts (MoE) architectures.
- Understand continuous batching and inference scheduling.

### 5. Alignment and evaluation

- Understand RLHF.
- Study preference-optimization methods such as DPO and GRPO.
- Learn LLM evaluation methodology and common benchmarks.
- Explore reasoning models.
- Study safety techniques and guardrails.

## Current focus

The current focus is foundations and pretraining: building and training a small
GPT-style model from scratch with PyTorch. Preserve that focus unless the user
explicitly asks to move to another part of the roadmap.

## How to collaborate in this repository

- Treat the repository as a learning environment. Explain the reasoning behind
  meaningful implementation choices, not only the final code.
- Match explanations to the user's current context. Define a new concept when it
  first appears, then use its standard terminology consistently.
- When introducing tensor operations, state the important input and output
  shapes and explain non-obvious dimension changes.
- Connect mathematical ideas to their PyTorch implementation when useful.
- Prefer small, inspectable steps and controlled experiments that isolate one
  idea at a time.
- Ask the user to make a decision or attempt an implementation when that creates
  genuine learning value. Do not turn every routine edit into an exercise.
- Do not hide core learning concepts behind a library abstraction before their
  from-scratch version is understood.
- It is acceptable to use high-level libraries for comparison, validation, or
  later roadmap stages; make that purpose explicit.
- Point out misconceptions directly and constructively. Distinguish facts,
  conventions, design choices, and open questions.
- Avoid adding unrelated architecture, dependencies, or roadmap scope.

## Repository workflow

Follow the established learning loop:

1. Explore a concept in a focused notebook or small experiment.
2. Inspect tensor shapes, intermediate values, behavior, and results.
3. Compare the result with a simple expectation or trusted implementation when
   practical.
4. Move code into `src/` only after it is understood, stable, and reusable.
5. Record the conclusion of an experiment, including what changed and what was
   learned.

Notebooks should tell a coherent learning story. Reusable modules should remain
clean enough to import without depending on notebook state.

## Implementation guidelines

- Use Python 3.12, PyTorch, JupyterLab, and `uv`, consistent with the existing
  project configuration.
- Prefer readable, explicit implementations over clever or premature
  optimizations.
- Keep experiments deterministic where practical: record seeds, configurations,
  dataset choices, and relevant environment assumptions.
- Make model and training configurations explicit rather than scattering magic
  numbers through the code.
- Add comments for intent, mathematical meaning, or surprising behavior; do not
  narrate obvious syntax.
- Preserve existing naming and project structure unless a change has a clear
  learning or maintenance benefit.
- Do not commit datasets, checkpoints, generated artifacts, or large model files
  unless the user explicitly requests it and the repository is configured for
  them.

## Verification

Scale verification to the change:

- For tensor or model code, check shapes, dtypes, devices, and a small forward
  pass.
- For training code, run the smallest useful smoke test and confirm that loss
  and gradients behave plausibly.
- For numerical reimplementations, compare against a reference or a simple
  hand-checkable case when possible.
- For notebooks, ensure cells run in order from a fresh kernel when practical.
- Report what was actually verified and identify any expensive training run that
  was intentionally not executed.

## Definition of done

A learning unit is complete when the implementation works at the intended
scale, the important behavior has been verified, and the repository captures
the main lesson well enough to revisit later. A passing run without an
understood result is not sufficient.
