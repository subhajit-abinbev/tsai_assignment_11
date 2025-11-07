import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any, Set

# Optional: If you install the `regex` package you can use grapheme clusters.
try:
    import regex as regx
    HAS_REGEX = True
except ImportError:
    HAS_REGEX = False

BENGALI_RANGE = ("\u0980", "\u09FF")
ZERO_WIDTH = {"\u200c", "\u200d"}

def normalize_bengali(text: str,
                      map_digits: bool = False,
                      keep_latin: bool = True,
                      collapse_punct: bool = True) -> str:
    """
    Basic Bengali normalization.
    - NFC normalize
    - Remove zero-width joiners
    - Optionally map Bengali digits to ASCII
    - Collapse excessive whitespace and punctuation repetition
    - Keep or drop Latin sequences (loanwords)
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Remove zero width markers
    text = "".join(ch for ch in text if ch not in ZERO_WIDTH)

    # Map Bengali digits
    if map_digits:
        digit_map = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
        text = text.translate(digit_map)

    # Optionally remove Latin (set to False if you want pure Bengali)
    if not keep_latin:
        text = re.sub(r"[A-Za-z]+", " ", text)

    # Collapse repeated danda/punctuations
    if collapse_punct:
        text = re.sub(r"।{2,}", "।", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

    # Remove URLs and markup fragments
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)

    # Keep Bengali block, ASCII digits, basic punctuation, Latin (if chosen)
    allowed = re.compile(r"[^\u0980-\u09FF0-9A-Za-z\s\.,!?;ঃ%“”\"'()।\-–—]")
    text = allowed.sub(" ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class BPEResult:
    vocab: Dict[str, int]
    merges: List[Tuple[str, str]]
    token2id: Dict[str, int]
    id2token: Dict[int, str]
    config: Dict[str, Any]


class CustomBPE:
    """
    Enhanced BPE implementation optimized for Bengali.
    Features:
    - Incremental pair frequency updates
    - Dropout for merge selection (diversifies subword units)
    - Grapheme-aware fallback splitting
    - Configurable normalization pipeline
    - Fragmentation and coverage reporting
    """

    def __init__(
        self,
        vocab_size: int = 12000,
        min_freq: int = 3,
        progress_every: int = 200,
        special_tokens: Optional[List[str]] = None,
        lowercase: bool = False,
        normalize_fn: Optional[Callable[[str], str]] = None,
        dropout: float = 0.0,
        max_word_length: Optional[int] = 100,
        early_stop_ratio: Optional[float] = None,
        enable_grapheme_fallback: bool = True,
        debug: bool = False,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self._word_freq: Counter = Counter()
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.progress_every = progress_every
        self.special_tokens = list(dict.fromkeys(special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
        self.lowercase = lowercase
        self.normalize_fn = normalize_fn
        self.dropout = dropout
        self.max_word_length = max_word_length
        self.early_stop_ratio = early_stop_ratio
        self.enable_grapheme_fallback = enable_grapheme_fallback
        self.debug = debug

        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.unk_token = "[UNK]"
        self.unk_token_id: Optional[int] = None

        # Internal structures
        self._word_tokens: Dict[str, List[str]] = {}
        self._pair_freq: Counter = Counter()
        self._affected_words: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # maps pair -> words containing it
        self._merge_cache_hits = 0

    @staticmethod
    def _word_to_symbols(word: str) -> List[str]:
        return list(word) + ["</w>"]

    def _preprocess_line(self, text: str) -> str:
        if self.normalize_fn:
            text = self.normalize_fn(text)
        if self.lowercase:
            text = text.lower()
        return text

    def _init_word_tokens(self) -> None:
        self._pair_freq.clear()
        self._affected_words.clear()
        for word, freq in self._word_freq.items():
            tokens = self._word_to_symbols(word)
            self._word_tokens[word] = tokens
            # update pair frequencies
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                self._pair_freq[pair] += freq
                self._affected_words[pair].add(word)

    def _select_best_pair(self) -> Optional[Tuple[str, str]]:
        if not self._pair_freq:
            return None
        # Apply dropout (ignore some high-frequency pairs randomly)
        candidates = []
        for pair, f in self._pair_freq.items():
            if f >= self.min_freq:
                if self.dropout > 0.0:
                    # With probability dropout, skip this pair in selection pool
                    import random
                    if random.random() < self.dropout:
                        continue
                candidates.append((pair, f))
        if not candidates:
            return None
        best_pair, best_freq = max(candidates, key=lambda x: x[1])
        return best_pair

    def _apply_merge(self, pair: Tuple[str, str]) -> None:
        """Merge a pair across all affected words, updating pair frequencies incrementally."""
        first, second = pair
        affected_words = self._affected_words.get(pair, set())
        if not affected_words:
            return

        # Prepare new pair_freq & affected mapping updates
        # We'll remove old pairs present in affected words and add new ones from merged tokens
        new_pair_freq_additions: Counter = Counter()
        new_affected_updates: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for word in affected_words:
            tokens = self._word_tokens[word]
            freq = self._word_freq[word]
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_token = first + second
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            # Remove old pair frequencies (approx: we'll recompute new ones; cheap for affected subset)
            self._word_tokens[word] = new_tokens

        # Rebuild pair frequencies ONLY for affected words
        # First, subtract old pair contributions
        for pair_old in list(self._pair_freq.keys()):
            # Only bother if pair_old had affected words & intersects our modified words
            if self._affected_words[pair_old] & affected_words:
                # We will fully recompute its frequency among changed words
                subtract_freq = 0
                for w in self._affected_words[pair_old] & affected_words:
                    # If the pair no longer appears in the word tokens after merge:
                    toks = self._word_tokens[w]
                    cnt = 0
                    for i in range(len(toks) - 1):
                        if (toks[i], toks[i + 1]) == pair_old:
                            cnt += self._word_freq[w]
                    # We will later add updated count
                    prev_count_estimate = self._word_freq[w] if cnt == 0 else 0
                # Instead of subtracting precisely, we will recompute entire pair_freq at end for changed words
                # Mark for cleanup later (cheap approach)
                pass

        # Clean old pair frequencies for changed words
        # Simplify: Recompute all pairs for changed words and adjust global counts.
        # First remove all contributions from changed words:
        for pair_key in list(self._pair_freq.keys()):
            if self._affected_words[pair_key] & affected_words:
                # Recalculate new total excluding changed words, then add updated counts
                old_words = self._affected_words[pair_key]
                changed = old_words & affected_words
                unchanged = old_words - affected_words
                # frequency contributed by unchanged words remains
                freq_new_total = 0
                for w in unchanged:
                    toks = self._word_tokens[w]
                    freq_w = self._word_freq[w]
                    for i in range(len(toks) - 1):
                        if (toks[i], toks[i + 1]) == pair_key:
                            freq_new_total += freq_w
                if freq_new_total > 0:
                    self._pair_freq[pair_key] = freq_new_total
                    self._affected_words[pair_key] = unchanged
                else:
                    del self._pair_freq[pair_key]
                    del self._affected_words[pair_key]

        # Now add new pairs for changed words
        for w in affected_words:
            toks = self._word_tokens[w]
            freq_w = self._word_freq[w]
            for i in range(len(toks) - 1):
                p = (toks[i], toks[i + 1])
                self._pair_freq[p] += freq_w
                self._affected_words[p].add(w)

        # Register merge
        self.merges.append(pair)
        self.merge_ranks[pair] = len(self.merges)

    def train(self, text_iter: Iterable[str]) -> BPEResult:
        # Build word frequency
        for line in text_iter:
            line = self._preprocess_line(line)
            if not line:
                continue
            for word in line.strip().split():
                if self.max_word_length and len(word) > self.max_word_length:
                    continue
                self._word_freq[word] += 1

        # Filter by min_freq
        self._word_freq = Counter({w: f for w, f in self._word_freq.items() if f >= self.min_freq})
        if not self._word_freq:
            raise ValueError("No words meet frequency threshold")

        self._init_word_tokens()

        target_vocab = self.vocab_size
        while len(self.special_tokens) + len(self.merges) < target_vocab:
            best_pair = self._select_best_pair()
            if not best_pair:
                if self.debug:
                    print("[CustomBPE] No valid pair above min_freq; stopping.")
                break
            self._apply_merge(best_pair)

            step = len(self.merges)
            if self.progress_every and step % self.progress_every == 0:
                if self.debug:
                    print(f"[CustomBPE] Merge {step}: {best_pair} | current vocab est={len(self.special_tokens)+len(self.merges)}")

            # Optional early stopping if fragmentation improves sufficiently
            if self.early_stop_ratio:
                frag = self.estimate_fragmentation(sample=5000)
                if frag <= self.early_stop_ratio:
                    if self.debug:
                        print(f"[CustomBPE] Early stop: fragmentation {frag:.2f} <= {self.early_stop_ratio}")
                    break

            if len(self.special_tokens) + len(self.merges) >= target_vocab:
                break

        # Build vocab counts
        vocab_counter = Counter()
        for word, tokens in self._word_tokens.items():
            freq = self._word_freq[word]
            for t in tokens:
                vocab_counter[t] += freq

        ordered = []
        for tok in self.special_tokens:
            if tok not in ordered:
                ordered.append(tok)
        for tok, _ in vocab_counter.most_common():
            if tok not in ordered:
                ordered.append(tok)
        if self.unk_token not in ordered:
            ordered.insert(1, self.unk_token)  # after [PAD]
        self.token2id = {tok: i for i, tok in enumerate(ordered)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self.unk_token_id = self.token2id[self.unk_token]

        config = {
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "progress_every": self.progress_every,
            "special_tokens": self.special_tokens,
            "lowercase": self.lowercase,
            "dropout": self.dropout,
            "max_word_length": self.max_word_length,
            "enable_grapheme_fallback": self.enable_grapheme_fallback,
        }

        return BPEResult(
            vocab=vocab_counter,
            merges=self.merges,
            token2id=self.token2id,
            id2token=self.id2token,
            config=config,
        )

    def _encode_word_tokens(self, word: str) -> List[str]:
        if not word:
            return []
        # Start with character symbols + boundary
        tokens = list(word) + ["</w>"]
        # Greedy merge application
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            best_pair = None
            best_rank = math.inf
            for p in pairs:
                r = self.merge_ranks.get(p)
                if r is not None and r < best_rank:
                    best_rank = r
                    best_pair = p
            if best_pair is None:
                break
            # Apply merge
            merged = []
            i = 0
            f, s = best_pair
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == f and tokens[i + 1] == s:
                    merged.append(f + s)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tokens

    def encode_with_tokens(self, text: str) -> Tuple[List[str], List[int]]:
        text = self._preprocess_line(text)
        if not text:
            return [], []
        tokens_out: List[str] = []
        ids_out: List[int] = []
        unk_id = self.unk_token_id if self.unk_token_id is not None else 0
        for word in text.strip().split():
            merged_tokens = self._encode_word_tokens(word)
            for m in merged_tokens:
                tok_id = self.token2id.get(m, None)
                if tok_id is None:
                    # Fallback splitting (grapheme clusters if available)
                    if self.enable_grapheme_fallback and HAS_REGEX:
                        for g in regx.findall(r"\X", m):
                            gid = self.token2id.get(g, unk_id)
                            tokens_out.append(g)
                            ids_out.append(gid)
                    else:
                        for ch in m:
                            cid = self.token2id.get(ch, unk_id)
                            tokens_out.append(ch)
                            ids_out.append(cid)
                else:
                    tokens_out.append(m)
                    ids_out.append(tok_id)
        return tokens_out, ids_out

    def encode(self, text: str) -> List[int]:
        return self.encode_with_tokens(text)[1]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2token.get(i, self.unk_token) for i in ids]
        words: List[str] = []
        current: List[str] = []
        for t in tokens:
            if t.endswith("</w>"):
                current.append(t[:-4])
                words.append("".join(current))
                current = []
            else:
                current.append(t)
        if current:
            words.append("".join(current))
        return " ".join(w for w in words if w)

    def estimate_fragmentation(self, sample: int = 10000) -> float:
        """Average number of tokens per original word (lower is better)."""
        total_tokens = 0
        total_words = 0
        for w, freq in self._word_freq.items():
            if total_words >= sample:
                break
            toks = self._encode_word_tokens(w)
            total_tokens += len(toks)
            total_words += 1
        return total_tokens / max(1, total_words)

    def char_coverage(self) -> Dict[str, float]:
        """Percent of corpus characters that appear in multi-char tokens (merged forms)."""
        total_chars = 0
        merged_chars = 0
        for w, freq in self._word_freq.items():
            total_chars += len(w) * freq
            toks = self._encode_word_tokens(w)
            for t in toks:
                if not t.endswith("</w>") and len(t) > 1:
                    merged_chars += len(t) * freq
        return {
            "total_chars": total_chars,
            "merged_chars": merged_chars,
            "multi_char_coverage_pct": (merged_chars / total_chars * 100) if total_chars else 0.0
        }

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "config": {
                "vocab_size": self.vocab_size,
                "min_freq": self.min_freq,
                "progress_every": self.progress_every,
                "special_tokens": self.special_tokens,
                "lowercase": self.lowercase,
                "dropout": self.dropout,
                "unk_token": self.unk_token,
                "enable_grapheme_fallback": self.enable_grapheme_fallback,
            },
            "merges": [list(pair) for pair in self.merges],
            "token2id": self.token2id,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_serializable(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CustomBPE":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        inst = cls(
            vocab_size=cfg.get("vocab_size", 8000),
            min_freq=cfg.get("min_freq", 2),
            progress_every=cfg.get("progress_every", 100),
            special_tokens=cfg.get("special_tokens", []),
            lowercase=cfg.get("lowercase", False),
            dropout=cfg.get("dropout", 0.0),
            enable_grapheme_fallback=cfg.get("enable_grapheme_fallback", True),
        )
        inst.unk_token = cfg.get("unk_token", inst.unk_token)
        inst.merges = [tuple(p) for p in data.get("merges", [])]
        inst.merge_ranks = {tuple(p): i for i, p in enumerate(inst.merges)}
        inst.token2id = {tok: int(i) for tok, i in data.get("token2id", {}).items()}
        inst.id2token = {i: tok for tok, i in inst.token2id.items()}
        inst.unk_token_id = inst.token2id.get(inst.unk_token)
        return inst