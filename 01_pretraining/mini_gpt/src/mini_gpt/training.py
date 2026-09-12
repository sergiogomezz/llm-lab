import torch
import torch.nn.functional as F

from .generation import generate_text_simple, text_to_token_ids, token_ids_to_text


def calc_loss_batch(input_batch, target_batch, model, device):
    """Return mean next-token cross-entropy for a batch of shape (B, T)."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    # Each position becomes one vocabulary classification: (B, T, V) -> (B*T, V).
    return F.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Average batch losses; use equal-size batches for a token-weighted mean.

    Gradient tracking and model mode are controlled by the caller.
    """
    if num_batches is not None and num_batches <= 0:
        raise ValueError("num_batches must be positive or None")

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Measure both splits without dropout or gradients, then resume train mode."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
        )
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model, tokenizer, device, start_context, training_context_length,
):
    """Print 50 greedy tokens using at most the context length used in training."""
    model.eval()
    context_length = min(
        training_context_length,
        model.positional_embedding.weight.shape[0],
    )
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            token_ids=encoded,
            max_new_tokens=50,
            context_length=context_length,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    training_context_length,
):
    """Train and return sampled losses and token counts for this call.

    Calling again continues from the current model and optimizer states,
    but restarts the reporting counters. Evaluation starts after the first update.
    """
    if eval_freq <= 0:
        raise ValueError("eval_freq must be positive")
    if eval_iter <= 0:
        raise ValueError("eval_iter must be positive")
    if training_context_length <= 0:
        raise ValueError("training_context_length must be positive")

    train_losses = []
    val_losses = []
    tokens_seen = []
    total_tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_seen.append(total_tokens_seen)
                print(
                    f"Epoch {epoch + 1} (step {global_step:06d}): "
                    f"train loss {train_loss:.3f}, "
                    f"validation loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context, training_context_length,
        )

    return train_losses, val_losses, tokens_seen
