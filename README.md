import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

@dataclass
class BPEResult:
    vocab: Dict[str, int]
    merges: List[Tuple[str, str]]
    token2id: Dict[str, int]
    id2token: Dict[int, str]

class CustomBPE:
    """
    Educational (not heavily optimized) BPE for demonstration.
    Operates on whitespace-delimited words and splits them into characters first.
    """
    def __init__(self, vocab_size: int = 8000, min_freq: int = 2, progress_every: int = 100):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.progress_every = progress_every
        self.merges = []
        self.token2id = {}
        self.id2token = {}

    @staticmethod
    def _word_to_symbols(word: str) -> List[str]:
        # Append </w> to mark word boundary (classic BPE style)
        return list(word) + ["</w>"]

    def _get_stats(self, corpus_tokens: List[List[str]]) -> Counter:
        stats = Counter()
        for word_tokens in corpus_tokens:
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i+1])
                stats[pair] += 1
        return stats

    def _merge_corpus(self, corpus_tokens: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        pattern = re.escape(pair[0]) + r"\s+" + re.escape(pair[1])
        repl = pair[0] + pair[1]
        new_corpus = []
        for tokens in corpus_tokens:
            if not tokens:
                new_corpus.append(tokens)
                continue
            s = " ".join(tokens)
            # Use regex to merge occurrences
            new_s = re.sub(pattern, repl, s)
            new_corpus.append(new_s.split(" "))
        return new_corpus

    def train(self, text_iter: Iterable[str]) -> BPEResult:
        # Build initial corpus word list
        vocab_counter = Counter()
        for line in text_iter:
            for word in line.strip().split():
                if word:
                    vocab_counter[word] += 1

        # Convert to list of token sequences
        corpus_tokens = []
        for word, freq in vocab_counter.items():
            if freq >= self.min_freq:
                symbols = self._word_to_symbols(word)
                # replicate frequency logically by storing frequency count
                corpus_tokens.extend([symbols] * min(freq, 5))  # small cap to control explosion
        # Iterative merges
        for step in range(self.vocab_size):
            stats = self._get_stats(corpus_tokens)
            if not stats:
                break
            pair, freq = stats.most_common(1)[0]
            if freq < self.min_freq:
                break
            corpus_tokens = self._merge_corpus(corpus_tokens, pair)
            self.merges.append(pair)
            if (step + 1) % self.progress_every == 0:
                print(f"[CustomBPE] Merge {step+1}: {pair} freq={freq} | merges so far={len(self.merges)}")

            if len(self.base_vocab()) + len(self.merges) >= self.vocab_size:
                break

        # Build final vocab
        final_vocab = Counter()
        for tokens in corpus_tokens:
            for t in tokens:
                final_vocab[t] += 1

        sorted_vocab = [tok for tok, _ in final_vocab.most_common()]
        self.token2id = {tok: i for i, tok in enumerate(sorted_vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        return BPEResult(
            vocab=final_vocab,
            merges=self.merges,
            token2id=self.token2id,
            id2token=self.id2token
        )

    def base_vocab(self):
        # characters + </w>
        chars = set()
        for pair in self.merges:
            for p in pair:
                for c in p:
                    chars.add(c)
        chars.add("</w>")
        return chars

    def encode_word(self, word: str) -> List[str]:
        # Greedy apply merges
        tokens = self._word_to_symbols(word)
        merges_as_strings = {"".join(p): p for p in self.merges}
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(tokens) - 1:
                pair = tokens[i] + tokens[i+1]
                if pair in merges_as_strings:
                    tokens = tokens[:i] + [pair] + tokens[i+2:]
                    merged = True
                else:
                    i += 1
        return tokens

    def encode(self, text: str) -> List[int]:
        result_ids = []
        for word in text.strip().split():
            for tok in self.encode_word(word):
                if tok in self.token2id:
                    result_ids.append(self.token2id[tok])
                else:
                    # Fallback to chars
                    for ch in tok:
                        result_ids.append(self.token2id.get(ch, self.token2id.get("</w>", 0)))
        return result_ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2token.get(i, "") for i in ids]
        words = []
        current = []
        for t in tokens:
            if t.endswith("</w>"):
                current.append(t.replace("</w>", ""))
                words.append("".join(current))
                current = []
            else:
                current.append(t)
        if current:
            words.append("".join(current))
        return " ".join(words)