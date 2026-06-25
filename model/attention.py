import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_layer = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self.key_layer(embedded)
        Q = self.query_layer(embedded)
        V = self.value_layer(embedded)

        attention_dim = K.shape[-1]
        scores = Q @ K.transpose(1, 2) / (attention_dim ** 0.5)

        context_length = embedded.shape[1]
        mask = torch.tril(torch.ones(context_length, context_length))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = torch.softmax(scores, dim=2)

        output = scores @ V
        return torch.round(output, decimals=4)