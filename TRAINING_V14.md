# TRAINING V14

The active corpus for Intelligence v14 is `dataset_v006`.

Build/refresh the deterministic intelligence packs and active splits:

```bash
python training/build_intelligence_v14.py
python training/tokenize_dataset_v14.py
```

Quality requirements enforced by the build:

- exactly 100 Generalization shards and 100 Reasoning shards;
- unique-input ratio >= 99.5%;
- unique-output ratio >= 99.5% (current release: 100% / 100%);
- zero cross-split `concept_group` leakage;
- deterministic/project-authored labels, no web-sourced labels;
- old v13 template-heavy packs are not inherited into active v006.

LAB now prepares `dataset_v006` before a training run and merges enabled user-authored supervised JSONL rows into the tokenized corpus. Tool-only LAB rows are intentionally skipped by language-model tokenization.

The bundled active neural checkpoint remains v13 because no new transformer checkpoint is promoted without a completed quality gate. Runtime v14 improvements work independently of that checkpoint through Router + ReasoningEngine + LocalIntelligenceV14.
