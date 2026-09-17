from __future__ import annotations

import base64
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = (
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<system>", "<user>", "<assistant>", "<tool>",
)


def normalize_language_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    translation = str.maketrans(
        {
            "ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "هٔ", "ة": "ه",
            "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا",
            "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
            "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        }
    )
    value = value.translate(translation)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.replace("\u200f", "").replace("\u200e", "")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


@dataclass(frozen=True, slots=True)
class TokenizerMetadata:
    format: str
    dataset_version: str
    vocab_size: int
    normalization: str
    pretrained_source: None = None


class JarvisTokenizer:
    """A JARVIS-trained byte-level BPE tokenizer with no external vocabulary."""

    FORMAT = "jarvis-byte-bpe-v2"
    LEGACY_FORMAT = "jarvis-byte-bpe-v1"
    _PRETOKEN = re.compile(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+|"
        r"[A-Za-z]:[\\/][^\s<>\"'|]+|"
        r"(?:[A-Za-z_][A-Za-z0-9_]*|[\u0600-\u06ff‌]+|\d+(?:\.\d+)?)|"
        r"\s+|[^\w\s]",
        re.UNICODE | re.I,
    )

    def __init__(
        self,
        token_bytes: list[bytes],
        merges: list[tuple[int, int]],
        dataset_version: str,
        boundary_aware: bool = False,
    ) -> None:
        if len(token_bytes) < len(SPECIAL_TOKENS) + 256:
            raise ValueError("Tokenizer is missing byte tokens")
        self.token_bytes = token_bytes
        self.merges = merges
        self.dataset_version = dataset_version
        self.boundary_aware = bool(boundary_aware)
        self.special_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.merge_rank = {pair: index for index, pair in enumerate(merges)}
        self.merge_to_id = {
            pair: len(SPECIAL_TOKENS) + 256 + index for index, pair in enumerate(merges)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<eos>"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        normalize: bool = True,
    ) -> list[int]:
        value = normalize_language_text(text) if normalize else text
        segments = self._PRETOKEN.findall(value) if self.boundary_aware else [value]
        tokens: list[int] = []
        for segment in segments:
            tokens.extend(self._encode_segment(segment))
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def _encode_segment(self, value: str) -> list[int]:
        data = value.encode("utf-8", errors="replace")
        tokens = [len(SPECIAL_TOKENS) + byte for byte in data]
        while len(tokens) > 1:
            best_index = -1
            best_rank = len(self.merges) + 1
            for index in range(len(tokens) - 1):
                rank = self.merge_rank.get((tokens[index], tokens[index + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_index = index
            if best_index < 0:
                break
            pair = (tokens[best_index], tokens[best_index + 1])
            tokens[best_index : best_index + 2] = [self.merge_to_id[pair]]
        return tokens

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        output = bytearray()
        for token_id in token_ids:
            index = int(token_id)
            if index < len(SPECIAL_TOKENS):
                if not skip_special:
                    output.extend(SPECIAL_TOKENS[index].encode("utf-8"))
                continue
            if 0 <= index < len(self.token_bytes):
                output.extend(self.token_bytes[index])
        return output.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT,
            "dataset_version": self.dataset_version,
            "normalization": "NFKC+PersianCanonical+LatinDigits",
            "pretokenizer": "jarvis_paths_urls_code_v2" if self.boundary_aware else "none",
            "special_tokens": list(SPECIAL_TOKENS),
            "token_bytes_b64": [
                base64.b64encode(value).decode("ascii") for value in self.token_bytes
            ],
            "merges": [[left, right] for left, right in self.merges],
            "pretrained_source": None,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "JarvisTokenizer":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") not in {cls.FORMAT, cls.LEGACY_FORMAT}:
            raise ValueError("Unsupported JARVIS tokenizer format")
        if tuple(payload.get("special_tokens", ())) != SPECIAL_TOKENS:
            raise ValueError("Tokenizer special token layout does not match runtime")
        token_bytes = [
            base64.b64decode(value) for value in payload["token_bytes_b64"]
        ]
        merges = [(int(left), int(right)) for left, right in payload["merges"]]
        return cls(
            token_bytes, merges, str(payload["dataset_version"]),
            boundary_aware=payload.get("pretokenizer") == "jarvis_paths_urls_code_v2",
        )

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int,
        dataset_version: str,
        minimum_pair_frequency: int = 2,
    ) -> "JarvisTokenizer":
        base_size = len(SPECIAL_TOKENS) + 256
        if vocab_size < base_size:
            raise ValueError(f"vocab_size must be at least {base_size}")
        sequence_counts: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            if not text.strip():
                continue
            normalized = normalize_language_text(text)
            for segment in cls._PRETOKEN.findall(normalized):
                sequence = tuple(
                    len(SPECIAL_TOKENS) + byte for byte in segment.encode("utf-8")
                )
                if sequence:
                    sequence_counts[sequence] += 1
        if not sequence_counts:
            raise ValueError("Cannot train tokenizer on an empty corpus")
        token_bytes = [b""] * len(SPECIAL_TOKENS) + [bytes((value,)) for value in range(256)]
        merges: list[tuple[int, int]] = []
        while len(token_bytes) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence, weight in sequence_counts.items():
                for pair in zip(sequence, sequence[1:]):
                    counts[pair] += weight
            if not counts:
                break
            pair, frequency = min(
                counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
            )
            if frequency < minimum_pair_frequency and len(token_bytes) >= base_size + 128:
                break
            new_id = len(token_bytes)
            token_bytes.append(token_bytes[pair[0]] + token_bytes[pair[1]])
            merges.append(pair)
            merged_counts: Counter[tuple[int, ...]] = Counter()
            for immutable, weight in sequence_counts.items():
                sequence = list(immutable)
                merged: list[int] = []
                index = 0
                while index < len(sequence):
                    if (
                        index + 1 < len(sequence)
                        and sequence[index] == pair[0]
                        and sequence[index + 1] == pair[1]
                    ):
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(sequence[index])
                        index += 1
                merged_counts[tuple(merged)] += weight
            sequence_counts = merged_counts
        return cls(token_bytes, merges, dataset_version, boundary_aware=True)
