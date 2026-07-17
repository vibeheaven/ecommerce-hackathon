"""
Batch Builder — Length-bucket batching to minimize padding.
Groups query-product pairs by token length, yielding massive inference speedups on GPU.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer

class PairInferenceDataset(Dataset):
    """Dataset for query-product pairs for fast inference."""

    def __init__(self, queries: list[str], titles: list[str], categories: list[str] | None = None):
        self.queries = queries
        self.titles = titles
        self.categories = categories if categories is not None else [""] * len(queries)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {
            "query": self.queries[idx],
            "title": self.titles[idx],
            "category": self.categories[idx],
        }


class LengthBucketSampler(Sampler):
    """
    Groups indices of similar sequence length together to minimize padding in batches.
    """

    def __init__(
        self,
        dataset: PairInferenceDataset,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_length: int = 256,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        # Precompute sequence lengths to sort
        lengths = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            # Construct text representation exactly as passed to the model
            text = f"{item['query']} | {item['title']}"
            if item["category"]:
                text += f" | {item['category']}"
            # Fast approximate length based on word split or short tokenization
            lengths.append(len(tokenizer.encode(text, max_length=max_length, truncation=True)))

        self.lengths = np.array(lengths)

    def __iter__(self):
        # Sort indices by length
        sorted_indices = np.argsort(self.lengths)

        # Chunk into batches
        batches = [
            sorted_indices[i : i + self.batch_size]
            for i in range(0, len(sorted_indices), self.batch_size)
        ]

        # Convert to list of indices
        flat_indices = []
        for batch in batches:
            flat_indices.extend(batch)

        return iter(flat_indices)

    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn_builder(tokenizer: PreTrainedTokenizer, max_length: int = 256):
    """Returns collate function that pads dynamically to the longest item in the batch."""
    def collate_fn(batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        texts = []
        for item in batch:
            text = f"{item['query']} | {item['title']}"
            if item.get("category"):
                text += f" | {item['category']}"
            texts.append(text)

        # Tokenize and pad dynamically to max length in this batch
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded

    return collate_fn
