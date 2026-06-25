import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V
        q = self.q_proj(x)  # (B, T, num_heads * head_dim)
        k = self.k_proj(x)  # (B, T, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, num_kv_heads * head_dim)

        # 2. Reshape into heads: (B, T, heads, head_dim) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Expand K, V to match num_heads by repeating each KV head
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)  # (B, num_heads, T, head_dim)
        v = v.repeat_interleave(repeats, dim=1)

        # 4. Scaled dot-product attention with causal mask
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, num_heads, T, T)

        causal_mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, num_heads, T, head_dim)

        # 5. Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.output_proj(out)

        return torch.round(out, decimals=4)