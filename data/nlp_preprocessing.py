import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        all_sentences = positive + negative

        # Build vocabulary: unique words across all sentences, sorted lexicographically
        vocab = set()
        for sentence in all_sentences:
            for word in sentence.split():
                vocab.add(word)

        sorted_vocab = sorted(vocab)
        word_to_id = {word: idx + 1 for idx, word in enumerate(sorted_vocab)}

        # Encode each sentence as a tensor of word IDs
        encoded = []
        for sentence in all_sentences:
            ids = [word_to_id[word] for word in sentence.split()]
            encoded.append(torch.tensor(ids, dtype=torch.float32))

        # Pad to rectangular shape (2N, T)
        padded = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=0)

        return padded