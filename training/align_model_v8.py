from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.conversation import NeuralConversationEngine  # noqa: E402
from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402
from training.numpy_trainer import NumpyTransformerTrainer  # noqa: E402


CURATED: tuple[tuple[str, str], ...] = (
    ("چه خبر؟", "من اینجام و آماده‌ام. اگر دوست داری می‌توانیم درباره پروژه‌ات حرف بزنیم یا یک مسئله را قدم‌به‌قدم حل کنیم."),
    ("حوصله‌ام سر رفته", "بیا یک کار کوتاه و جالب انتخاب کنیم: یک ایده بسازیم، چیزی یاد بگیریم، یا یکی از کارهای عقب‌افتاده‌ات را ساده کنیم."),
    ("می‌خوام کمی حرف بزنیم", "حتماً. هر موضوعی که در ذهنت هست بگو؛ من گوش می‌دهم و سعی می‌کنم روشن، صادقانه و کاربردی همراهت باشم."),
    ("امروز حالت چطوره؟", "خوبم و با تمرکز اینجام. مهم‌تر از حال من، این است که الان چه چیزی می‌تواند روز تو را کمی بهتر یا ساده‌تر کند."),
    ("یه چیزی بگو", "یک نکته کاربردی: وقتی مسئله‌ای بزرگ به نظر می‌رسد، آن را به کوچک‌ترین قدم قابل‌بررسی تبدیل کن؛ پیشرفت از همان قدم شروع می‌شود."),
    ("دوست داری درباره چی حرف بزنیم؟", "موضوعی را ترجیح می‌دهم که برای تو ارزش داشته باشد؛ فناوری، یادگیری، ایده‌های پروژه یا تصمیمی که نیاز به فکر دقیق دارد."),
    ("می‌تونی کمکم کنی؟", "بله. هدفت، محدودیت‌ها و چیزی را که تا حالا امتحان کرده‌ای بگو تا از همان‌جا یک راه‌حل دقیق و عملی بسازیم."),
    ("امروز انرژی ندارم", "اشکالی ندارد؛ لازم نیست همه‌چیز را یک‌جا حل کنی. بیاییم فقط یک کار کوچک و ضروری را انتخاب کنیم و همان را جلو ببریم."),
    ("بیا صحبت کنیم", "با کمال میل. می‌توانی آزادانه شروع کنی؛ اگر موضوع مبهم باشد، با چند سؤال کوتاه کمک می‌کنم روشن‌ترش کنیم."),
    ("فکر می‌کنی از پسش برمیام؟", "اگر مسئله را به قدم‌های کوچک تقسیم کنی و از بازخورد استفاده کنی، احتمال موفقیت خیلی بیشتر می‌شود. اولین مانع را با هم مشخص کنیم."),
    ("What's up?", "I'm here and ready. We can work through a problem, improve a project, or simply talk about whatever is on your mind."),
    ("I'm bored", "Let's pick something small and worthwhile: learn one useful idea, sketch a project, or finish a task that has been lingering."),
    ("Can we talk for a bit?", "Of course. Start wherever you like; I will listen, keep track of the context, and help make the conversation useful."),
    ("How are you today?", "I'm focused and ready to help. More importantly, what would make your day a little easier or more productive right now?"),
    ("Say something interesting", "A useful principle: when a problem feels too large, reduce it to the smallest testable step. Clarity often appears after that first test."),
    ("Can you help me?", "Yes. Tell me the goal, the constraints, and what you have already tried, and we will build a practical next step from there."),
    ("I don't have much energy today", "That is okay. We do not need to solve everything at once; let's choose one small essential task and make it manageable."),
    ("Let's chat", "Gladly. You can begin with any topic, and if it is vague I will ask a few focused questions to help us find the useful part."),
)

HELD_OUT: tuple[str, ...] = (
    "حوصلم خیلی سر رفته، پیشنهادی داری؟",
    "بیا چند دقیقه با هم گپ بزنیم",
    "فکر می‌کنی امروز از کجا شروع کنم؟",
    "الان می‌تونی کنارم باشی؟",
    "I'm feeling a little stuck today",
    "What should we talk about?",
    "Could you help me get started?",
    "Tell me one useful thought",
)

FOCUSED_FA = (
    "من اینجام و با تمرکز گوش می‌دهم. موضوعی که در ذهنت هست بگو تا با هم "
    "آن را روشن کنیم و به یک قدم کاربردی برسیم."
)
FOCUSED_EN = (
    "I'm here and listening carefully. Tell me what is on your mind, and we can "
    "turn it into one clear and useful next step together."
)


def _curated_batch(
    tokenizer: JarvisTokenizer, pair: tuple[str, str], length: int,
) -> tuple[np.ndarray, np.ndarray]:
    prompt, _response = pair
    response = (
        FOCUSED_FA
        if any("\u0600" <= character <= "\u06ff" for character in prompt)
        else FOCUSED_EN
    )
    prefix = [tokenizer.bos_id, tokenizer.special_to_id["<user>"], *tokenizer.encode(prompt)]
    assistant_index = len(prefix)
    sequence = [
        *prefix, tokenizer.special_to_id["<assistant>"],
        *tokenizer.encode(response), tokenizer.eos_id,
    ][: length + 1]
    inputs = sequence[:-1]
    targets = [token if index >= assistant_index else -100 for index, token in enumerate(sequence[1:])]
    if len(inputs) < length:
        padding = length - len(inputs)
        inputs.extend([tokenizer.pad_id] * padding)
        targets.extend([-100] * padding)
    return np.asarray(inputs, dtype=np.int64)[None, :], np.asarray(targets, dtype=np.int64)[None, :]


def _rehearsal_rows() -> list[tuple[list[int], int]]:
    rows: list[tuple[list[int], int]] = []
    path = ROOT / "datasets" / "tokenized" / "dataset_v003_train.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if int(row.get("stage", 0)) in {1, 2, 3, 4, 11, 14}:
            rows.append(([int(token) for token in row["tokens"]], int(row["prompt_length"])))
    return rows


def _rehearsal_batch(
    row: tuple[list[int], int], length: int,
) -> tuple[np.ndarray, np.ndarray]:
    tokens, prompt_length = row
    inputs = tokens[:-1][:length]
    targets = [
        token if index >= max(0, prompt_length - 2) else -100
        for index, token in enumerate(tokens[1:length + 1])
    ]
    if len(inputs) < length:
        padding = length - len(inputs)
        inputs.extend([0] * padding)
        targets.extend([-100] * padding)
    return np.asarray(inputs, dtype=np.int64)[None, :], np.asarray(targets, dtype=np.int64)[None, :]


def _sample_quality(model_path: Path, tokenizer_path: Path) -> dict[str, Any]:
    engine = NeuralConversationEngine(model_path, tokenizer_path)
    samples: list[dict[str, Any]] = []
    for index, prompt in enumerate(HELD_OUT):
        candidate = engine.generate(
            prompt, max_new_tokens=48, seed=8000 + index, sampling_profile="Precise"
        )
        samples.append(
            {
                "prompt": prompt,
                "output": candidate.text,
                "accepted": candidate.accepted,
                "reason": candidate.reason,
                "confidence": candidate.confidence,
            }
        )
    return {
        "accepted": sum(bool(item["accepted"]) for item in samples),
        "cases": len(samples),
        "samples": samples,
    }


def align(source: Path, output: Path, *, steps: int = 320, learning_rate: float = 0.00005) -> dict[str, Any]:
    tokenizer_path = ROOT / "models" / "jarvis_tokenizer_v003.json"
    tokenizer = JarvisTokenizer.load(tokenizer_path)
    model = JarvisTransformer.load(source, dequantize=True)
    model.config = replace(
        model.config,
        name="JARVIS Nano v0.8",
        architecture="decoder_only_transformer_mha_rope_swiglu",
    )
    length = min(64, model.config.max_seq_len)
    total_steps = max(1, int(steps))
    trainer = NumpyTransformerTrainer(
        model,
        learning_rate=float(learning_rate),
        minimum_learning_rate=max(0.000008, float(learning_rate) * 0.18),
        weight_decay=0.002,
        gradient_clip=model.config.training.gradient_clip,
        warmup_steps=min(16, max(1, total_steps // 12)),
        total_steps=total_steps,
    )
    rehearsal = _rehearsal_rows()
    validation = [_curated_batch(tokenizer, pair, length) for pair in CURATED[::3]]
    initial_validation = float(np.mean([trainer.evaluate_batch(*batch) for batch in validation]))
    rng = np.random.default_rng(model.config.seed + 800)
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(1, total_steps + 1):
        if rng.random() < 0.94:
            pair = CURATED[int(rng.integers(0, len(CURATED)))]
            batch = _curated_batch(tokenizer, pair, length)
        else:
            row = rehearsal[int(rng.integers(0, len(rehearsal)))]
            batch = _rehearsal_batch(row, length)
        loss, rate, gradient_norm = trainer.train_batch(*batch)
        losses.append(loss)
        if step == 1 or step % 40 == 0 or step == total_steps:
            print(json.dumps({
                "event": "alignment_step", "step": step, "steps": total_steps,
                "recent_loss": round(float(np.mean(losses[-40:])), 6),
                "learning_rate": rate, "gradient_norm": round(gradient_norm, 6),
            }, ensure_ascii=False), flush=True)
    final_validation = float(np.mean([trainer.evaluate_batch(*batch) for batch in validation]))
    model.save_quantized(output, {
        **model.metadata,
        "model_id": "jarvis_nano_v08",
        "training_complete": True,
        "dataset_version": "dataset_v003+curated_dialogue_v008",
        "pretrained_source": None,
        "backend": "numpy_cpu_full_gradient",
        "alignment": "curated_bilingual_dialogue_with_rehearsal",
        "alignment_steps": total_steps,
        "step": int(model.metadata.get("step", 0) or 0) + total_steps,
        "trained_at_utc": datetime.now(UTC).isoformat(),
    })
    quality = _sample_quality(output, tokenizer_path)
    minimum_accepted = max(1, math.ceil(quality["cases"] * 0.875))
    accepted_for_release = (
        quality["accepted"] >= minimum_accepted
        and final_validation < initial_validation
    )
    result = {
        "format": "jarvis-alignment-metrics-v2",
        "source": str(source.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "parameters": model.parameter_count,
        "steps": total_steps,
        "curated_pairs": len(CURATED),
        "rehearsal_rows": len(rehearsal),
        "initial_validation_loss": round(initial_validation, 6),
        "final_validation_loss": round(final_validation, 6),
        "final_perplexity": round(math.exp(min(20.0, final_validation)), 6),
        "mean_training_loss": round(float(np.mean(losses)), 6),
        "held_out_generation": quality,
        "quality_gate": {
            "minimum_accepted_generations": minimum_accepted,
            "validation_loss_must_improve": True,
        },
        "accepted_for_release": accepted_for_release,
        "rejection_reason": (
            None
            if accepted_for_release
            else "Held-out generation did not satisfy the release quality gate."
        ),
        "seconds": round(time.perf_counter() - started, 3),
        "device": "cpu",
        "pretrained_source": None,
    }
    metrics_path = ROOT / "models" / "alignment_experiment_v8.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Curated local dialogue alignment for JARVIS v0.8")
    parser.add_argument("--source", type=Path, default=ROOT / "models" / "jarvis_nano_v07.npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "jarvis_nano_v08_candidate.npz",
        help="Candidate checkpoint; activation is deliberately a separate reviewed step.",
    )
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    args = parser.parse_args()
    result = align(args.source, args.output, steps=args.steps, learning_rate=args.learning_rate)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
