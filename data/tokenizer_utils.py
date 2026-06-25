from typing import List, Dict

class Solution:
    def _greedy_tokenize(self, s: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(s)

        while i < n:
            longest_match = None
            # Try longest possible substring first, shrinking down to length 1
            for j in range(n, i, -1):
                candidate = s[i:j]
                if candidate in vocab:
                    longest_match = candidate
                    break

            if longest_match is not None:
                tokens.append(longest_match)
                i += len(longest_match)
            else:
                # No match found at all (even length 1) - consume single character
                tokens.append(s[i])
                i += 1

        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self._greedy_tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()
        word_count = len(words)
        token_count = len(self._greedy_tokenize(text, vocab))
        fertility = token_count / word_count
        return round(fertility, 4)