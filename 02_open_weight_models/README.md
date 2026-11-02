# Open-weight models: Qwen3

Completed the Qwen3 learning unit: from Hub artifacts to local inference and a
controlled comparison of base and post-trained behavior. This closes the focused
Qwen3 chapter, not every architecture or topic in the wider roadmap.

## Learning path

1. [Hub and checkpoint](qwen3/notebooks/01_hub_and_checkpoint.ipynb): distinguish
   configuration, tokenizer and learned weights; inspect tensors and pin revisions.
2. [Tokenizer and forward](qwen3/notebooks/02_tokenizer_and_forward.ipynb): connect
   text, token IDs and next-token logits to the mini GPT implementation.
3. [Text generation](qwen3/notebooks/03_text_generation.ipynb): implement decoding,
   explore sampling and inspect KV caching. The recorded MPS/BF16 timing experiment
   reduced median time from 0.9902 s to 0.3903 s (2.54x). Cached and uncached
   continuations differed; each manual loop matched its corresponding library path.
4. [Base versus post-trained](qwen3/notebooks/04_base_vs_posttrained.ipynb): compare
   five fixed tasks, control input formatting and inspect thinking under a fixed budget.

## Final experiment: observed results

Notebook 04 was executed from a fresh kernel on 2026-11-02 using cached, pinned
Qwen3-0.6B-Base and Qwen3-0.6B checkpoints, Apple MPS and BF16. All nine code cells
completed without errors. Both models have 596,049,920 parameters and use about
1.110 GiB for their BF16 parameters. Models were loaded one at a time and released.

| Observation | Lesson |
| --- | --- |
| Base and post-trained plain-text runs all reached the 64-token cap without EOS; post-trained chat emitted EOS on four of five tasks. | A learned conversational format helps orient generation toward bounded assistant responses. |
| The same post-trained model answered Canberra in both formats, but plain text continued with more questions. | Formatting affects continuation behavior without changing the weights or removing knowledge. |
| Chat respected the two-sentence instruction but returned fenced JSON with `creator: None`. | Better instruction following is not perfect adherence or factual reliability. |
| Thinking used 155 tokens versus 11 for the capital question and added an inaccurate location claim. | More generated thinking does not guarantee a better answer. |
| Thinking exhausted 256 tokens on the JSON task before reaching a final answer. | The output budget includes thinking tokens; distinguish unfinished reasoning from a final response. |

The main base/chat comparison changes both weights and formatting. The separate
post-trained plain/chat control holds weights and decoding fixed, measuring the
combined effect of the template, including its non-thinking prefix. Five tasks,
greedy decoding and short budgets do not establish a general benchmark or isolate
a specific post-training algorithm's effect.

## Connection to mini GPT and the next stage

The mechanism remains token IDs → model → logits → token selection → repeat.
Hub tooling packages the components; Transformers lets us inspect a real model;
KV caching avoids recomputing past keys and values. Post-training changes weights,
chat templates format inputs, and decoding selects outputs. These are distinct
parts of the same inference process.

The next planned stage is fine-tuning: changing pretrained weights with training
examples. No fine-tuning experiments were started as part of this chapter closure.
