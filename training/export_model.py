from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.torch_model import JarvisTorchTransformer, torch_available  # noqa: E402
from jarvis.neural.transformer import JarvisTransformer  # noqa: E402


def _convert(source: JarvisTransformer) -> object:
    if not torch_available():
        raise RuntimeError("PyTorch export needs requirements-training.txt")
    import torch

    target = JarvisTorchTransformer(source.config)
    state = target.state_dict()
    mapping: dict[str, object] = {
        "tok_embeddings.weight": source.parameters["tok_embeddings"],
        "lm_head.weight": (
            source.parameters["tok_embeddings"]
            if source.config.tie_embeddings else source.parameters["lm_head"].T
        ),
        "final_norm.weight": source.parameters["final_norm"],
    }
    for layer in range(source.config.n_layers):
        prefix = f"layers.{layer}"
        mapping.update(
            {
                f"{prefix}.attn_norm.weight": source.parameters[f"{prefix}.attn_norm"],
                f"{prefix}.attention.wq.weight": source.parameters[f"{prefix}.wq"].T,
                f"{prefix}.attention.wk.weight": source.parameters[f"{prefix}.wk"].T,
                f"{prefix}.attention.wv.weight": source.parameters[f"{prefix}.wv"].T,
                f"{prefix}.attention.wo.weight": source.parameters[f"{prefix}.wo"].T,
                f"{prefix}.ffn_norm.weight": source.parameters[f"{prefix}.ffn_norm"],
                f"{prefix}.feed_forward.w1.weight": source.parameters[f"{prefix}.w1"].T,
                f"{prefix}.feed_forward.w2.weight": source.parameters[f"{prefix}.w2"].T,
                f"{prefix}.feed_forward.w3.weight": source.parameters[f"{prefix}.w3"].T,
            }
        )
    for name, value in mapping.items():
        state[name] = torch.from_numpy(value.copy())  # type: ignore[union-attr]
    target.load_state_dict(state, strict=True)
    target.eval()
    return target


def export(model_path: Path, output: Path, kind: str, precision: str) -> dict[str, object]:
    import torch

    source = JarvisTransformer.load(model_path, dequantize=True)
    model = _convert(source)
    if precision == "fp16":
        model = model.half()  # type: ignore[union-attr]
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "jarvis-export-v1", "source": str(model_path),
        "architecture": source.config.architecture, "parameters": source.parameter_count,
        "dataset_version": source.metadata.get("dataset_version"),
        "pretrained_source": None, "precision": precision, "kind": kind,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if kind == "torch":
        torch.save(
            {**metadata, "config": source.config.to_dict(), "model_state": model.state_dict()},
            output,
        )
    else:
        try:
            import onnx  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ONNX export needs requirements-training.txt") from exc

        class Wrapper(torch.nn.Module):
            def __init__(self, inner: object) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                return self.inner(input_ids)["logits"]

        dummy = torch.tensor([[1, 5, 6]], dtype=torch.long)
        torch.onnx.export(
            Wrapper(model), dummy, output, input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
            opset_version=17,
        )
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return {**metadata, "output": str(output), "size_bytes": output.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export only project-owned JARVIS weights")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "jarvis_nano_v07.npz")
    parser.add_argument("--format", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suffix = ".pt" if args.format == "torch" else ".onnx"
    output = args.output or ROOT / "models" / "exports" / f"jarvis_nano_{args.precision}{suffix}"
    print(json.dumps(export(args.model, output, args.format, args.precision), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
