"""Dataset and data-loader utilities for next-token prediction."""

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    """Create overlapping input-target token sequences from text."""

    def __init__(self, text, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(text)

        self.input_ids = []
        self.target_ids = []

        for index in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[index : index + max_length]
            target_chunk = token_ids[index + 1 : index + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_data_loader_v1(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """Build a data loader containing shifted input-target token batches."""

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
