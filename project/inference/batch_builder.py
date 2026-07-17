"""
Batch Builder — Length-bucket batching to minimize padding.
Groups query-product pairs by token length, yielding massive inference speedups on GPU.

The (query, product) texts must be built with project.utils.pair_text — the
same builder used in training — so there is no train/serve skew.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer


class PairInferenceDataset(Dataset):
    """Dataset of prebuilt (query, product_text) pairs for fast inference."""

    def __init__(self, queries: list[str], products: list[str]):
        assert len(queries) == len(products)
        self.queries = queries
        self.products = products

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {"query": self.queries[idx], "product": self.products[idx]}


class LengthBucketSampler(Sampler):
    """
    Groups indices of similar sequence length together to minimize padding in batches.
    """

    def __init__(
        self,
        dataset: PairInferenceDataset,
        batch_size: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        # Character length is a good-enough proxy for token length and avoids
        # tokenizing the whole dataset twice.
        self.lengths = np.array(
            [len(q) + len(p) for q, p in zip(dataset.queries, dataset.products)]
        )

    def __iter__(self):
        sorted_indices = np.argsort(self.lengths, kind="stable")
        return iter(sorted_indices.tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class InferenceCollator:
    """Picklable collate: pads dynamically to the longest item in the batch."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            [item["query"] for item in batch],
            [item["product"] for item in batch],
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )


def collate_fn_builder(tokenizer: PreTrainedTokenizer, max_length: int = 256) -> InferenceCollator:
    return InferenceCollator(tokenizer, max_length)
