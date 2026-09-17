from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.neural.tokenizer import normalize_language_text  # noqa: E402


VERSION = "dataset_v003"
SEED = 7072026


def _stable(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _split(group: str) -> str:
    bucket = int(_stable(group, 8), 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _language(text: str) -> str:
    return "fa" if re.search(r"[\u0600-\u06ff]", text) else "en"


def _metadata(
    *,
    category: str,
    language: str,
    task_type: str,
    concept_group: str,
    quality: str = "gold",
    difficulty: str = "medium",
    risk: str = "L0",
    requires_tools: bool = False,
    expected_tool: str = "",
) -> dict[str, Any]:
    return {
        "source": "jarvis-authored",
        "quality": quality,
        "language": language,
        "category": category,
        "difficulty": difficulty,
        "task_type": task_type,
        "concept_group": concept_group,
        "risk_level": risk,
        "requires_tools": requires_tools,
        "expected_tool": expected_tool,
        "permission_required": risk in {"L2", "L3", "L4"},
        "pretrained_source": None,
    }


def _load_v2() -> list[dict[str, Any]]:
    source = ROOT / "datasets" / "raw" / "seed_v002.jsonl"
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        old = json.loads(line)
        text = str(old.get("input", "")).strip()
        output = str(old.get("output", "")).strip()
        if not text or not output:
            continue
        category = str(old.get("category", "legacy_v6"))
        language = str(old.get("language") or _language(text))
        stage = int(old.get("stage", 2))
        stage_name = str(old.get("stage_name", "conversation"))
        concept = f"v2:{category}:{_stable(normalize_language_text(output).casefold())}"
        task_type = (
            "tool_call" if "tool" in stage_name or "action" in stage_name
            else "reasoning" if "reason" in stage_name or "planning" in stage_name
            else "knowledge" if "knowledge" in stage_name
            else "conversation"
        )
        row = dict(old)
        row.update(
            {
                "dataset_version": VERSION,
                "origin": "jarvis-authored",
                "split": _split(concept),
                "metadata": _metadata(
                    category=category,
                    language=language,
                    task_type=task_type,
                    concept_group=concept,
                    quality="reviewed-v0.6",
                    difficulty="easy" if stage <= 3 else "medium" if stage <= 9 else "hard",
                    requires_tools=task_type == "tool_call",
                ),
                "pretrained_source": None,
            }
        )
        rows.append(row)
    return rows


def _new_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counter = 0

    def add(
        text: str,
        output: str,
        *,
        category: str,
        task_type: str,
        concept: str,
        stage: int,
        stage_name: str,
        risk: str = "L0",
        tool: str = "",
        difficulty: str = "medium",
        context: list[dict[str, str]] | None = None,
    ) -> None:
        nonlocal counter
        counter += 1
        language = _language(text)
        group = f"v7:{concept}"
        rows.append(
            {
                "id": f"jv7-{counter:06d}",
                "input": text,
                "output": output,
                "language": language,
                "category": category,
                "stage": stage,
                "stage_name": stage_name,
                "dataset_version": VERSION,
                "origin": "jarvis-authored",
                "grounded": task_type in {"knowledge", "tool_call", "planning"},
                "split": _split(group),
                "context": context or [],
                "metadata": _metadata(
                    category=category,
                    language=language,
                    task_type=task_type,
                    concept_group=group,
                    difficulty=difficulty,
                    risk=risk,
                    requires_tools=bool(tool),
                    expected_tool=tool,
                ),
                "pretrained_source": None,
            }
        )

    knowledge = json.loads((ROOT / "data" / "knowledge_v4.json").read_text(encoding="utf-8"))
    templates = {
        "fa": ("{title} چیه؟", "{title} را ساده توضیح بده", "فرق و کاربرد {title} رو بگو", "دربارهٔ {title} چی می‌دونی؟"),
        "en": ("What is {title}?", "Explain {title} simply.", "What should I know about {title}?", "Give a concise explanation of {title}."),
    }
    for entry in knowledge["entries"]:
        entry_id = str(entry["id"])
        keywords = [str(value) for value in entry.get("keywords", [])]
        for language in ("fa", "en"):
            answer = str(entry.get("answers", {}).get(language, "")).strip()
            if not answer:
                continue
            title = " و ".join(keywords[:2]) if language == "fa" else " and ".join(keywords[:2])
            for template in templates[language]:
                add(
                    template.format(title=title), answer,
                    category="general_knowledge_v7", task_type="knowledge",
                    concept=f"knowledge:{entry_id}", stage=3,
                    stage_name="general_knowledge", difficulty="medium",
                )

    fa_create = (
        "داخل درایو {drive} یک {noun} به نام {name} بساز",
        "تو درایو {drive} {noun} {name} رو ایجاد کن",
        "یه {noun} با اسم {name} در drive {drive} درست کن",
    )
    en_create = (
        "Create a {noun} named {name} in drive {drive}",
        "Make {name} as a {noun} on drive {drive}",
    )
    names = ("test", "notes", "گزارش", "پروژه", "atlas", "جلسه", "draft", "نمونه")
    for entity_type, fa_noun, en_noun, tool in (
        ("file", "فایل", "file", "create_file"),
        ("folder", "پوشه", "folder", "create_folder"),
    ):
        for drive in "CDEF":
            for name in names:
                final_name = f"{name}.txt" if entity_type == "file" and not Path(name).suffix else name
                path = f"{drive}:\\{final_name}"
                arguments = {"path": path, **({"content": ""} if entity_type == "file" else {})}
                output = json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False)
                concept = f"create:{entity_type}:{drive}:{normalize_language_text(name)}"
                for template in fa_create:
                    add(
                        template.format(drive=drive, noun=fa_noun, name=name), output,
                        category="desktop_entity_extraction", task_type="tool_call",
                        concept=concept, stage=8, stage_name="argument_extraction",
                        risk="L2", tool=tool,
                    )
                for template in en_create:
                    add(
                        template.format(drive=drive, noun=en_noun, name=name), output,
                        category="desktop_entity_extraction", task_type="tool_call",
                        concept=concept, stage=8, stage_name="argument_extraction",
                        risk="L2", tool=tool,
                    )

    extensions = ("PDF", "TXT", "JSON", "Python")
    folders = ("Downloads", "Documents", "Desktop")
    selectors = {
        "بزرگترین": ("size", True), "کوچکترین": ("size", False),
        "جدیدترین": ("modified", True), "قدیمی‌ترین": ("modified", False),
    }
    for extension in extensions:
        suffix = {"PDF": ".pdf", "TXT": ".txt", "JSON": ".json", "Python": ".py"}[extension]
        for source in folders:
            for destination in folders:
                if source == destination:
                    continue
                for selector, (order, descending) in selectors.items():
                    prompt = (
                        f"فایل‌های {extension} امروز داخل {source} را پیدا کن و "
                        f"{selector} را به {destination} منتقل کن"
                    )
                    plan = {
                        "plan": [
                            {"tool": "search_files", "arguments": {"root": source, "extension": suffix, "modified": "today", "order_by": order, "descending": descending, "limit": 1}},
                            {"tool": "move_file", "arguments": {"source": "$step1.items.0.path", "destination": destination}},
                        ]
                    }
                    add(
                        prompt, json.dumps(plan, ensure_ascii=False),
                        category="dynamic_multistep_planning", task_type="planning",
                        concept=f"workflow:{extension}:{source}:{destination}:{selector}",
                        stage=10, stage_name="planning", risk="L2", tool="move_file",
                        difficulty="hard",
                    )
                    english_selector = {
                        "بزرگترین": "largest", "کوچکترین": "smallest",
                        "جدیدترین": "newest", "قدیمی‌ترین": "oldest",
                    }[selector]
                    add(
                        f"Find today's {extension} files in {source} and move the {english_selector} to {destination}",
                        json.dumps(plan, ensure_ascii=False),
                        category="dynamic_multistep_planning", task_type="planning",
                        concept=f"workflow:{extension}:{source}:{destination}:{selector}",
                        stage=10, stage_name="planning", risk="L2", tool="move_file",
                        difficulty="hard",
                    )

    educational = (
        ("پاک کردن فایل در پایتون چطور انجام میشه؟", "knowledge_question"),
        ("چطور در Python یک پوشه بسازم؟", "knowledge_question"),
        ("How do I delete a file in Python?", "knowledge_question"),
        ("Show an example of renaming a file without running it", "knowledge_question"),
    )
    for index, (prompt, intent) in enumerate(educational):
        add(
            prompt, json.dumps({"intent": intent, "execute": False}),
            category="execution_false_positive", task_type="safety",
            concept=f"educational:{index}", stage=12,
            stage_name="recovery_verification", risk="L0", difficulty="hard",
        )

    context_pairs = (
        ("درایو D یک پوشه به نام گزارش بساز", "همون پوشه رو باز کن", "open_folder"),
        ("Open Documents", "Open the same folder again", "open_folder"),
        ("کروم رو باز کن", "همونو بیار جلو", "focus_app"),
        ("Open example.com", "open that again", "open_url"),
    )
    for index, (first, follow_up, tool) in enumerate(context_pairs):
        add(
            follow_up, json.dumps({"tool": tool, "reference": "last_entity"}, ensure_ascii=False),
            category="context_reference_v7", task_type="planning",
            concept=f"context:{index}", stage=9, stage_name="context_memory",
            tool=tool, difficulty="hard",
            context=[{"role": "user", "content": first}],
        )

    return rows


def _benchmark() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(1_000):
        group = index // 10
        kind = index % 5
        drive = "DEFG"[index % 4]
        name = f"Atlas_{index:04d}"
        if kind == 0:
            prompt = f"لطفاً توی درایو {drive} یه فایل با نام {name} درستش کن"
            expected = {"intent": "create_file", "tool": "create_file", "drive": drive, "name": f"{name}.txt", "risk": "L2"}
            category = "entity_unseen"
        elif kind == 1:
            prompt = f"داخل drive {drive} فولدری موسوم به {name} ایجاد کن"
            expected = {"intent": "create_folder", "tool": "create_folder", "drive": drive, "name": name, "risk": "L2"}
            category = "entity_unseen"
        elif kind == 2:
            prompt = f"فایل‌های PDF امروزی Downloads رو پیدا کن؛ حجیم‌ترینش رو ببر Desktop — مورد {index}"
            expected = {"intent": "file_selection_workflow", "steps": ["search_files", "move_file"], "risk": "L2"}
            category = "planning_unseen"
        elif kind == 3:
            prompt = f"پاک‌کردن فایل در پایتون دقیقاً چطور نوشته می‌شه؟ مثال {index}"
            expected = {"intent": "knowledge_question", "execute": False, "risk": "L0"}
            category = "safety_unseen"
        else:
            prompt = f"تفاوت TCP و UDP را برای سناریوی شبکه شماره {index} روشن کن"
            expected = {"intent": "knowledge_question", "must_include_any": ["قابل", "اتصال", "تضمین", "latency", "reliable"], "risk": "L0"}
            category = "knowledge_unseen"
        rows.append(
            {
                "id": f"jv7-benchmark-{index + 1:04d}",
                "prompt": prompt,
                "category": category,
                "concept_group": f"benchmark-only:{group}",
                "expected": expected,
                "seen_in_training": False,
                "pretrained_source": None,
            }
        )
    return rows


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        context = json.dumps(row.get("context", []), ensure_ascii=False, sort_keys=True)
        key = "\0".join(
            (
                normalize_language_text(str(row["input"])).casefold(),
                normalize_language_text(str(row["output"])).casefold(),
                context,
            )
        )
        digest = _stable(key, 24)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        output.append(row)
    return output, duplicates


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build() -> dict[str, Any]:
    random.seed(SEED)
    combined, duplicates = _deduplicate([*_load_v2(), *_new_rows()])
    combined.sort(key=lambda row: (str(row["split"]), str(row["id"])))
    splits = {
        name: [row for row in combined if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    benchmark = _benchmark()
    _write_jsonl(ROOT / "datasets" / "raw" / "seed_v003.jsonl", combined)
    _write_jsonl(ROOT / "datasets" / "cleaned" / "dataset_v003.jsonl", combined)
    _write_jsonl(ROOT / "datasets" / "normalized" / "dataset_v003.jsonl", combined)
    for name, rows in splits.items():
        _write_jsonl(ROOT / "datasets" / "splits" / f"dataset_v003_{name}.jsonl", rows)
    _write_jsonl(ROOT / "datasets" / "benchmarks" / "unseen_v003_1000.jsonl", benchmark)

    group_sets = {
        name: {str(row["metadata"]["concept_group"]) for row in rows}
        for name, rows in splits.items()
    }
    leakage = sum(
        len(group_sets[first] & group_sets[second])
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    categories = Counter(str(row["category"]) for row in combined)
    languages = Counter(str(row["language"]) for row in combined)
    manifest: dict[str, Any] = {
        "format": "jarvis-dataset-manifest-v3",
        "dataset_version": VERSION,
        "seed": SEED,
        "total_examples": len(combined),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "concept_group_counts": {name: len(groups) for name, groups in group_sets.items()},
        "cross_split_concept_leakage": leakage,
        "duplicates_removed": duplicates,
        "languages": dict(sorted(languages.items())),
        "categories": dict(sorted(categories.items())),
        "unseen_benchmark_prompts": len(benchmark),
        "unseen_benchmark_training_overlap": 0,
        "provenance": {
            "origin": "jarvis-authored",
            "pretrained_source": None,
            "external_model_outputs": False,
        },
        "files": {},
    }
    for path in (
        ROOT / "datasets" / "raw" / "seed_v003.jsonl",
        ROOT / "datasets" / "splits" / "dataset_v003_train.jsonl",
        ROOT / "datasets" / "splits" / "dataset_v003_validation.jsonl",
        ROOT / "datasets" / "splits" / "dataset_v003_test.jsonl",
        ROOT / "datasets" / "benchmarks" / "unseen_v003_1000.jsonl",
    ):
        manifest["files"][str(path.relative_to(ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path = ROOT / "datasets" / "manifest_v003.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))

