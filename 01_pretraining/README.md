# Pretraining: GPT from scratch

Completed the core implementation and pretraining learning unit from *Build a
Large Language Model (From Scratch)* using PyTorch.

## What was covered

- Tokenization, input/target windows, data loaders and embeddings.
- Causal multi-head attention, normalization, feed-forward layers and residuals.
- A full GPT-style model and greedy autoregressive generation.
- Next-token cross-entropy, perplexity, AdamW and train/validation evaluation.
- Saving and restoring model weights and optimizer state.

## Experiment and conclusions

Trained on *The Verdict* for 10 epochs (90 optimizer updates), using a 90/10
text split, 256-token sequences and batches of 2 on Apple MPS.

Initial loss was about 10.99. The last recorded training-batch loss was 0.52;
validation loss reached its recorded minimum of 6.13 around epoch 6, then rose.
The model learned the tiny corpus and overfit it; this was an educational run,
not a general-purpose language model.

Reloaded parameters were identical, CPU/MPS logits agreed within numerical
tolerance, and the restored AdamW state retained all 90 updates.

## Files and next step

- [Notebooks](mini_gpt/notebooks/): data preparation, attention, architecture and pretraining.
- [Reusable code](mini_gpt/src/mini_gpt/): model, data, generation and training helpers.
- [Next chapter: open-weight models](../02_open_weight_models/qwen3/README.md).

Temperature/top-k sampling will be explored in the next chapter. Loading the
book's pretrained GPT-2 weights and its fine-tuning chapters remain deferred.
