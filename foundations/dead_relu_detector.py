import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []

        with torch.no_grad():
            out = x
            for layer in model.children():
                out = layer(out)
                if isinstance(layer, nn.ReLU):
                    dead = (out == 0).all(dim=0).float().mean().item()
                    dead_fractions.append(round(dead, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if not dead_fractions:
            return 'healthy'

        # 1. Severe: any layer > 0.5
        if any(f > 0.5 for f in dead_fractions):
            return 'use_leaky_relu'

        # 2. First layer > 0.3
        if dead_fractions[0] > 0.3:
            return 'reinitialize'

        # 3. Strictly increasing with depth AND last layer > 0.1
        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )
        if strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        # 4. Healthy: max < 0.1
        if max(dead_fractions) < 0.1:
            return 'healthy'

        # 5. Healthy: fallback
        return 'healthy'