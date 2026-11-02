# Fine-tuning

Status: next agreed stage; experiment not started.

Learn how training on a new dataset changes a pretrained model, building on the
base versus post-trained comparisons in the open-weight models stage.

## Resume here: fine-tuning for tool usage

Handoff recorded on 2026-11-02. The mini GPT unit and the Qwen3 chapter are
complete. Notebook 04 was executed from a fresh kernel and its results and
limitations are recorded in [the Qwen3 conclusions](../02_open_weight_models/README.md).

The user chose **tool usage as the adaptation experiment**. The objective is to
fine-tune an existing model to decide when a tool is needed, select the right tool
and produce a structured call with correct arguments. Include requests that
should be answered without a tool. Producing a tool call and executing the tool
are separate steps; start with small, local, deterministic tools.

When resuming:

1. Define a small task, a few tool schemas and representative requests together.
2. Choose a model and training method appropriate for the available hardware.
   Qwen3 is a familiar candidate, not a finalized choice; full fine-tuning versus
   LoRA remains to be decided.
3. Establish a baseline before training, using the intended tool-call format.
4. Build a small training set and a held-out evaluation set. Include variations
   in wording and arguments, plus cases requiring no tool; avoid train/test leakage.
5. Inspect how examples become tokens, training targets and loss masks, then run
   the smallest useful adaptation experiment.
6. Compare before/after tool selection, schema validity, argument correctness and
   unnecessary calls. Where useful, execute calls against local test tools to
   check their results. Record losses, trainable parameters and resource use.

No model, dataset, tool domain or training method has been selected. No training
has run. Start with the task definition and baseline rather than implementing
all future fine-tuning topics at once.

## Topics

- [ ] Learn the Hugging Face Transformers and Datasets workflows.
- [ ] Perform full fine-tuning and inspect which parameters receive gradients.
- [ ] Understand and implement low-rank adaptation (LoRA).
- [ ] Explore quantized LoRA (QLoRA) and its memory trade-offs.
- [ ] Perform instruction tuning, inspecting formatting and loss masking.

Start with a small, reproducible dataset and model. Compare behavior before and
after training; record trainable parameter counts, memory use, losses and held-out
results. Revisit the book's deferred fine-tuning chapters when useful.

Add notebooks as this stage begins, then extract understood, reusable code into
`src/`. Check off topics only after verification and recorded conclusions.

[Full roadmap](../README.md#learning-roadmap) · [Next: inference and deployment](../04_inference_and_deployment/README.md)
