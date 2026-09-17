from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.config import TransformerConfig  # noqa: E402
from jarvis.neural.tokenizer import JarvisTokenizer  # noqa: E402
from jarvis.neural.torch_model import JarvisTorchTransformer, torch_available  # noqa: E402


def _load_sequences(dataset_version: str) -> dict[int, list[tuple[list[int], int]]]:
    result = {stage: [] for stage in range(1, 15)}
    path = ROOT / "datasets" / "tokenized" / f"{dataset_version}_train.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Tokenized training dataset not found: {path}. Tokenize {dataset_version} first.")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result[int(row["stage"])].append(
            ([int(token) for token in row["tokens"]], int(row["prompt_length"]))
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train JARVIS Nano/Core/Pro from random initialization; no pretrained models"
    )
    parser.add_argument("--profile", choices=("nano", "core", "pro"), default="core")
    parser.add_argument("--steps-per-stage", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--dataset-version", default="dataset_v007")
    args = parser.parse_args()
    if not torch_available():
        parser.error("PyTorch is required. Run setup.ps1 -Training first.")
    import torch

    config_name = "nano_v7.json" if args.profile == "nano" else f"{args.profile}.json"
    config = TransformerConfig.load(ROOT / "configs" / config_name)
    tokenizer = JarvisTokenizer.load(ROOT / "models" / "jarvis_tokenizer_v003.json")
    if tokenizer.vocab_size != config.vocab_size:
        raise RuntimeError("Project tokenizer vocabulary does not match the selected profile")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JarvisTorchTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay, betas=(0.9, 0.95),
    )
    completed_steps = 0
    parent: str | None = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("pretrained_source") not in (None, ""):
            raise RuntimeError("External pretrained checkpoints are forbidden")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed_steps = int(checkpoint.get("step", 0))
        parent = str(args.resume)
    sequences = _load_sequences(args.dataset_version)
    sequence_length = min(
        args.sequence_length or config.training.sequence_length, config.max_seq_len
    )
    batch_size = args.batch_size or config.training.micro_batch_size
    accumulation = args.gradient_accumulation or config.training.gradient_accumulation
    steps_per_stage = args.steps_per_stage or config.training.steps_per_stage
    total_steps = steps_per_stage * 14
    randomizer = random.Random(config.seed + completed_steps)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = ROOT / "logs" / "training" / f"torch_{timestamp}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(value: dict[str, Any]) -> None:
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **value}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)

    def batch(stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for _ in range(batch_size):
            pieces: list[int] = []
            labels: list[int] = []
            while len(pieces) < sequence_length:
                selected = stage if stage == 1 or randomizer.random() < 0.72 else randomizer.randint(1, stage)
                tokens, prompt_length = randomizer.choice(sequences[selected])
                pieces.extend(tokens[:-1])
                labels.extend(
                    token if index >= max(0, prompt_length - 2) else -100
                    for index, token in enumerate(tokens[1:])
                )
            maximum_start = len(pieces) - sequence_length
            candidates = [
                start for start in range(maximum_start + 1)
                if any(value >= 0 for value in labels[start : start + sequence_length])
            ]
            start = randomizer.choice(candidates)
            input_rows.append(pieces[start : start + sequence_length])
            label_rows.append(labels[start : start + sequence_length])
        return (
            torch.tensor(input_rows, dtype=torch.long, device=device),
            torch.tensor(label_rows, dtype=torch.long, device=device),
        )

    def control_state() -> dict[str, bool]:
        if not args.control_file or not args.control_file.is_file():
            return {"pause": False, "stop": False, "save": False}
        try:
            value = json.loads(args.control_file.read_text(encoding="utf-8"))
            return {name: bool(value.get(name, False)) for name in ("pause", "stop", "save")}
        except (OSError, json.JSONDecodeError):
            return {"pause": False, "stop": False, "save": False}

    def clear_save_request() -> None:
        if not args.control_file:
            return
        state = control_state()
        state["save"] = False
        temporary = args.control_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(args.control_file)

    def save_checkpoint(path: Path, stage: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "jarvis-torch-checkpoint-v1",
                "config": config.to_dict(), "dataset_version": args.dataset_version,
                "pretrained_source": None, "training_stages": list(range(1, stage + 1)),
                "step": global_step, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            path,
        )

    use_amp = device.type == "cuda"
    precision = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == torch.float16)
    started = time.perf_counter()
    global_step = completed_steps
    emit(
        {
            "event": "training_start", "profile": config.profile,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "dataset_version": args.dataset_version, "pretrained_source": None,
            "device": str(device), "cuda": torch.cuda.is_available(),
            "precision": str(precision if use_amp else torch.float32),
            "checkpoint_parent": parent,
        }
    )
    for stage in range(1, 15):
        model.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for stage_step in range(1, steps_per_stage + 1):
            state = control_state()
            while state["pause"] and not state["stop"]:
                time.sleep(0.15)
                state = control_state()
            if state["stop"]:
                emergency = ROOT / "models" / "checkpoints" / f"torch_stage{stage}" / "stopped_checkpoint.pt"
                save_checkpoint(emergency, stage)
                emit({"event": "training_stopped", "stage": stage, "step": global_step, "checkpoint": str(emergency.relative_to(ROOT))})
                return 0
            if state["save"]:
                manual = ROOT / "models" / "checkpoints" / f"torch_stage{stage}" / f"manual_step{global_step}.pt"
                save_checkpoint(manual, stage)
                clear_save_request()
                emit({"event": "checkpoint_saved", "stage": stage, "step": global_step, "checkpoint": str(manual.relative_to(ROOT))})
            step_started = time.perf_counter()
            accumulated_loss = 0.0
            for _ in range(accumulation):
                input_ids, targets = batch(stage)
                with torch.autocast(device_type=device.type, dtype=precision, enabled=use_amp):
                    result = model(input_ids, targets)
                    loss = result["loss"] / accumulation
                scaler.scale(loss).backward()
                accumulated_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip
            )
            progress = global_step / max(1, total_steps)
            rate = config.training.minimum_learning_rate + (
                config.training.learning_rate - config.training.minimum_learning_rate
            ) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            for group in optimizer.param_groups:
                group["lr"] = rate
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            losses.append(accumulated_loss)
            duration = max(time.perf_counter() - step_started, 1e-9)
            emit(
                {
                    "event": "step", "stage": stage, "stage_step": stage_step,
                    "step": global_step, "loss": round(accumulated_loss, 6),
                    "learning_rate": rate, "gradient_norm": float(gradient_norm),
                    "tokens_per_second": round(
                        batch_size * accumulation * sequence_length / duration, 3
                    ),
                }
            )
        checkpoint_directory = ROOT / "models" / "checkpoints" / f"torch_stage{stage}"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_directory / "checkpoint.pt"
        save_checkpoint(checkpoint_path, stage)
        emit(
            {
                "event": "stage_complete", "stage": stage,
                "mean_loss": sum(losses) / len(losses),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            }
        )
    emit(
        {
            "event": "training_complete", "steps": global_step,
            "seconds": round(time.perf_counter() - started, 3),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
