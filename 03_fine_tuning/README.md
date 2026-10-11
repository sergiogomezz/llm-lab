# Fine-tuning

Status: planned.

Learn how training on a new dataset changes a pretrained model, building on the
base versus post-trained comparisons in the open-weight models stage.

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
