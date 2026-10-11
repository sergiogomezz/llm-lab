# LLM internals and optimization

Status: planned.

Connect components seen in open-weight models to small, inspectable PyTorch
implementations and controlled comparisons with the original mini GPT.

## Topics

- [ ] Study rotary positional embeddings (RoPE).
- [ ] Study root mean square normalization (RMSNorm) and SwiGLU feed-forward layers.
- [ ] Inspect KV-cache tensor shapes, growth and memory use in detail.
- [ ] Study multi-query and grouped-query attention (MQA/GQA).
- [ ] Understand FlashAttention and why it improves performance.
- [ ] Explore speculative decoding.
- [ ] Explore multi-token prediction (MTP).
- [ ] Study mixture-of-experts (MoE) architectures.
- [ ] Understand continuous batching and inference scheduling in depth.

Isolate one change at a time. Predict shapes and behavior, compare with a simple
reference, then measure performance where the available hardware allows it.

Add notebooks as this stage begins, then extract understood, reusable code into
`src/`. Check off topics only after verification and recorded conclusions.

[Full roadmap](../README.md#learning-roadmap) · [Next: alignment and evaluation](../06_alignment_and_evaluation/README.md)
