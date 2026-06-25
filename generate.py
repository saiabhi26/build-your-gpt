import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = ""
        for i in range(new_chars):

            # Crop context to the last context_length tokens
            cropped_context = context[:, -context_length:]

            # Forward pass, take logits at the last position
            logits = model(cropped_context)
            last_logits = logits[:, -1, :]

            # Convert to probabilities
            probs = torch.softmax(last_logits, dim=-1)

            # Reset RNG state, then sample
            generator.set_state(initial_state)
            next_token = torch.multinomial(probs, 1, generator=generator)

            # Append sampled token to context
            context = torch.cat([context, next_token], dim=1)

            # Map token id to character and accumulate
            result += int_to_char[next_token.item()]

        return result