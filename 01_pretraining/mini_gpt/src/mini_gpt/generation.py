"""Token conversion and greedy text-generation utilities."""

import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(
        text,
        allowed_special={"<|endoftext|>"},
    )
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat_token_ids = token_ids.squeeze(0)
    return tokenizer.decode(flat_token_ids.tolist())


def generate_text_simple(
    model,
    token_ids,
    max_new_tokens,
    context_length,
):
    """Generate tokens greedily, always selecting the largest final logit."""

    for _ in range(max_new_tokens):
        input_tokens = token_ids[:, -context_length:]

        with torch.no_grad():
            logits = model(input_tokens)

        last_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(
            last_token_logits,
            dim=-1,
            keepdim=True,
        )
        token_ids = torch.cat((token_ids, next_token_id), dim=1)

    return token_ids
