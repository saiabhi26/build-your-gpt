import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        positions = np.arange(seq_len).reshape(-1, 1)          # shape (seq_len, 1)
        i = np.arange(0, d_model, 2)                            # 0, 2, 4, ... shape (d_model/2,)
        div_term = 10000 ** (i / d_model)                       # shape (d_model/2,)

        angles = positions / div_term                           # broadcast -> (seq_len, d_model/2)

        PE = np.zeros((seq_len, d_model), dtype=np.float64)
        PE[:, 0::2] = np.sin(angles)
        PE[:, 1::2] = np.cos(angles)

        return np.round(PE, 5)