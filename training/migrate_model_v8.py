from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.transformer import JarvisTransformer


def migrate(source: Path, output: Path) -> None:
    """Correct legacy architecture metadata without changing quantized parameters."""
    model = JarvisTransformer.load(source, dequantize=False)
    model.config = replace(
        model.config,
        name="JARVIS Nano v0.8",
        architecture="decoder_only_transformer_mha_rope_swiglu",
    )
    model.save_quantized(
        output,
        {
            **model.metadata,
            "model_id": "jarvis_nano_v08",
            "metadata_migrated_from": source.name,
            "metadata_migrated_at_utc": datetime.now(UTC).isoformat(),
            "dialogue_adapter": "dialogue_adapter_v8.json",
            "pretrained_source": None,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate JARVIS v0.7 model metadata to v0.8")
    parser.add_argument("--source", type=Path, default=ROOT / "models" / "jarvis_nano_v07.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "jarvis_nano_v08.npz")
    args = parser.parse_args()
    migrate(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
