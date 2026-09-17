from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from jarvis.neural.tokenizer import JarvisTokenizer, normalize_language_text
from jarvis.neural.transformer import JarvisTransformer
from jarvis.utils.text import detect_language, normalize_text, tokenize


@dataclass(frozen=True, slots=True)
class NeuralCandidate:
    text: str
    accepted: bool
    reason: str
    generated_tokens: int
    latency_ms: float
    tokens_per_second: float
    confidence: float = 0.0
    sampling_profile: str = "Balanced"


@lru_cache(maxsize=2)
def _load_assets(
    model_path: str, tokenizer_path: str,
) -> tuple[JarvisTransformer, JarvisTokenizer]:
    model=JarvisTransformer.load(Path(model_path))
    selection=Path(model_path).parent/'language_adapter_v21.json'
    if selection.is_file():
        config=json.loads(selection.read_text())
        if config.get('enabled'):
            import hashlib
            if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()==config.get('base_sha256'):
                from jarvis.neural.language_adapter_v21 import attach_adapter
                model=attach_adapter(model,selection.parent/config['file'],Path(model_path))
    return model,JarvisTokenizer.load(Path(tokenizer_path))


class NeuralConversationEngine:
    """Quality-gated conversation generation using the JARVIS-owned LM."""

    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._model: JarvisTransformer | None = None
        self._tokenizer: JarvisTokenizer | None = None
        self._metadata = JarvisTransformer.peek_metadata(model_path)
        self._dialogue_examples = self._load_dialogue_adapter(
            model_path.parent / "dialogue_adapter_v8.json"
        )

    @staticmethod
    def _load_dialogue_adapter(path: Path) -> tuple[tuple[str, str, str], ...]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ()
        if payload.get("format") != "jarvis-dialogue-adapter-v1":
            return ()
        examples: list[tuple[str, str, str]] = []
        for item in payload.get("examples", []):
            if not isinstance(item, dict):
                continue
            prompt = normalize_text(str(item.get("prompt", "")))
            response = str(item.get("response", "")).strip()
            language = str(item.get("language", detect_language(prompt)))
            if prompt and response and language in {"fa", "en"}:
                examples.append((prompt, response, language))
        return tuple(examples[:200])

    @staticmethod
    def _dialogue_similarity(left: str, right: str) -> float:
        def ngrams(value: str) -> set[str]:
            compact = re.sub(r"\s+", " ", normalize_text(value)).strip()
            padded = f"  {compact}  "
            return {padded[index : index + 3] for index in range(max(0, len(padded) - 2))}

        left_grams, right_grams = ngrams(left), ngrams(right)
        dice = 2.0 * len(left_grams & right_grams) / max(
            1, len(left_grams) + len(right_grams)
        )
        left_words, right_words = set(tokenize(left)), set(tokenize(right))
        word_score = len(left_words & right_words) / max(1, len(left_words | right_words))
        return dice * 0.72 + word_score * 0.28

    def _adapter_candidate(self, text: str, profile_name: str) -> NeuralCandidate | None:
        language = detect_language(text)
        ranked = sorted(
            (
                (self._dialogue_similarity(text, prompt), response)
                for prompt, response, example_language in self._dialogue_examples
                if example_language == language
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.16:
            return None
        similarity, response = ranked[0]
        accepted, _reason, confidence = self._quality(response, text)
        if not accepted:
            return None
        return NeuralCandidate(
            response, True, "dialogue_adapter", 0, 0.0, 0.0,
            round(max(confidence, min(0.94, 0.68 + similarity * 0.28)), 4),
            f"{profile_name}/Adapter",
        )

    @property
    def model(self) -> JarvisTransformer:
        if self._model is None:
            self._model, self._tokenizer = _load_assets(
                str(self.model_path), str(self.tokenizer_path)
            )
        return self._model

    @property
    def tokenizer(self) -> JarvisTokenizer:
        if self._tokenizer is None:
            self._model, self._tokenizer = _load_assets(
                str(self.model_path), str(self.tokenizer_path)
            )
        return self._tokenizer

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def parameter_count(self) -> int:
        return int(self._metadata.get("parameter_count", 0))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @staticmethod
    def _quality(text: str, prompt: str = "") -> tuple[bool, str, float]:
        value = text.strip()
        if len(value) < 2:
            return False, "empty_or_too_short", 0.0
        lowered = value.casefold()
        if "�" in value or any(
            token in lowered
            for token in ("<tool", "<user", "<assistant", "<system", "</tool", "tool_call")
        ):
            return False, "invalid_or_structured_token", 0.0
        words = re.findall(r"[\w\u0600-\u06ff]+", value.casefold())
        if not words:
            return False, "no_language_content", 0.0
        if len(words) >= 5:
            dominant = max(words.count(word) for word in set(words))
            if dominant / len(words) > 0.42:
                return False, "repetition_guard", 0.12
            trigrams = list(zip(words, words[1:], words[2:]))
            if trigrams and len(set(trigrams)) / len(trigrams) < 0.62:
                return False, "duplicate_phrase_guard", 0.14
            common = {
                "و", "با", "را", "رو", "در", "به", "از", "که", "تا", "هم", "آن", "این",
                "and", "or", "the", "a", "an", "to", "of", "in", "on", "is", "are",
            }
            if sum(word in common for word in words) / len(words) > 0.52:
                return False, "low_information_word_distribution", 0.12
        if re.search(r"(.{2,12})\1\1\1", value):
            return False, "substring_repetition_guard", 0.12
        replacement_ratio = value.count("?") / max(1, len(value))
        if replacement_ratio > 0.12:
            return False, "decode_uncertainty", 0.18
        printable = sum(
            character.isprintable() or character == "\u200c" for character in value
        ) / max(1, len(value))
        language_chars = sum(
            character.isalpha() or character.isdigit() or character.isspace()
            or character in ".,!?؟،؛:;'-()[]\u200c"
            for character in value
        ) / max(1, len(value))
        if printable < 0.98 or language_chars < 0.82:
            return False, "invalid_character_distribution", 0.16

        normalized_prompt = normalize_text(prompt)
        normalized_output = normalize_text(value)
        if normalized_prompt and (
            normalized_output == normalized_prompt
            or (len(normalized_prompt) >= 8 and normalized_output.startswith(normalized_prompt + " "))
        ):
            return False, "prompt_echo_guard", 0.18
        prompt_words = {word for word in tokenize(normalized_prompt) if len(word) >= 3}
        output_words = {word for word in tokenize(normalized_output) if len(word) >= 3}
        relevance = (
            len(prompt_words & output_words) / max(1, min(len(prompt_words), 8))
            if prompt_words else 0.5
        )
        if normalized_prompt:
            prompt_language = detect_language(normalized_prompt)
            output_language = detect_language(normalized_output)
            strong_prompt = (
                len(re.findall(r"[\u0600-\u06ff]", normalized_prompt)) >= 4
                or len(re.findall(r"[a-z]", normalized_prompt)) >= 8
            )
            if strong_prompt and prompt_language != output_language and relevance == 0:
                return False, "language_consistency_guard", 0.2
        if len(words) >= 9 and re.search(
            r"(?:\b(?:and|or|because|but|that|to)\b|(?:و|یا|چون|اما|که|تا))\s*$",
            normalized_output,
            re.I,
        ):
            return False, "incomplete_sentence_guard", 0.22
        length_score = min(1.0, len(value) / 80.0)
        diversity = len(set(words)) / max(1, len(words))
        confidence = min(
            0.94,
            0.3 + 0.24 * length_score + 0.25 * diversity
            + 0.15 * min(1.0, relevance * 2.0),
        )
        accepted = confidence >= 0.64
        return accepted, "passed" if accepted else "low_generation_confidence", confidence

    def generate(
        self,
        text: str,
        *,
        recent_context: tuple[str, ...] = (),
        max_new_tokens: int = 48,
        seed: int | None = None,
        sampling_profile: str = "Balanced",
        cancel_check: Callable[[], bool] | None = None,
    ) -> NeuralCandidate:
        profile_name = str(sampling_profile).title()
        profiles = {
            "Precise": {"temperature": 0.16, "top_k": 12, "top_p": 0.72, "repetition_penalty": 1.16},
            "Balanced": {"temperature": 0.38, "top_k": 24, "top_p": 0.88, "repetition_penalty": 1.13},
            "Creative": {"temperature": 0.72, "top_k": 48, "top_p": 0.95, "repetition_penalty": 1.08},
        }
        sampling = profiles.get(profile_name, profiles["Balanced"])
        if profile_name not in profiles:
            profile_name = "Balanced"
        adapter = self._adapter_candidate(text, profile_name)
        if adapter is not None:
            return adapter
        tokenizer = self.tokenizer
        context = "\n".join(recent_context[-3:])
        prompt_text = f"{context}\n{text}".strip() if context else text
        prompt = [
            tokenizer.bos_id,
            tokenizer.special_to_id["<user>"],
            *tokenizer.encode(prompt_text),
            tokenizer.special_to_id["<assistant>"],
        ]
        prompt = prompt[-(self.model.config.max_seq_len - max_new_tokens) :]
        started = time.perf_counter()
        generated = self.model.sample(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=sampling["temperature"],
            top_k=int(sampling["top_k"]),
            top_p=sampling["top_p"],
            repetition_penalty=sampling["repetition_penalty"],
            eos_id=tokenizer.eos_id,
            seed=seed,
            should_cancel=cancel_check,
            suppress_token_ids=tuple(dict.fromkeys([
                *(
                    token_id for token, token_id in tokenizer.special_to_id.items()
                    if token not in {"<eos>", "<unk>"}
                ),
                *tokenizer.encode("<"),
            ])),
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        output = normalize_language_text(tokenizer.decode(generated)).strip()
        accepted, reason, confidence = self._quality(output, text)
        return NeuralCandidate(
            output,
            accepted,
            reason,
            len(generated),
            round(elapsed * 1000.0, 3),
            round(len(generated) / elapsed, 3),
            round(confidence, 4),
            profile_name,
        )

    def close(self) -> None:
        # Shared immutable weights stay cached for other sessions in this process.
        return None
