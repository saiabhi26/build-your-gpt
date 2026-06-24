import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x, dtype=np.float64)
        W1 = np.array(W1, dtype=np.float64)
        b1 = np.array(b1, dtype=np.float64)
        W2 = np.array(W2, dtype=np.float64)
        b2 = np.array(b2, dtype=np.float64)
        y_true = np.array(y_true, dtype=np.float64)

        n = len(y_true)

        # Forward pass
        z1 = x @ W1.T + b1          # (hidden_size,)
        a1 = np.maximum(0, z1)      # ReLU
        z2 = a1 @ W2.T + b2         # (output_size,)
        y_hat = z2

        # Loss
        loss = np.mean((y_hat - y_true) ** 2)

        # Backward pass
        dz2 = 2 * (y_hat - y_true) / n          # (output_size,)
        dW2 = np.outer(dz2, a1)                 # (output_size, hidden_size)
        db2 = dz2                               # (output_size,)

        da1 = W2.T @ dz2                        # (hidden_size,)
        relu_mask = (z1 > 0).astype(np.float64)
        dz1 = da1 * relu_mask                   # (hidden_size,)

        dW1 = np.outer(dz1, x)                  # (hidden_size, input_size)
        db1 = dz1                               # (hidden_size,)

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }