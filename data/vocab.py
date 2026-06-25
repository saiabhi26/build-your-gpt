from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        unique_chars = sorted(set(text))
        stoi = {char: idx for idx, char in enumerate(unique_chars)}
        itos = {idx: char for idx, char in enumerate(unique_chars)}
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        return [stoi[char] for char in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        return ''.join(itos[i] for i in ids)