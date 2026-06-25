import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)

        max_start = len(data) - context_length - 1
        start_indices = torch.randint(0, max_start + 1, (batch_size,))

        X = torch.stack([data[start:start + context_length] for start in start_indices])
        Y = torch.stack([data[start + 1:start + 1 + context_length] for start in start_indices])

        return X, Y