import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        h = np.array(x, dtype=np.float64)
        num_layers = len(weights)

        for i in range(num_layers):
            W = np.array(weights[i], dtype=np.float64)
            b = np.array(biases[i], dtype=np.float64)
            h = h @ W + b

            # Apply ReLU on every layer except the last (output layer)
            if i < num_layers - 1:
                h = np.maximum(0, h)

        return np.round(h, 5)