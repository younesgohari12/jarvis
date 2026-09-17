from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.agent.deliberation import TextReasoner  # noqa: E402
from jarvis.config import load_config  # noqa: E402
from jarvis.knowledge.store import KnowledgeStore  # noqa: E402
from jarvis.search.engine import SearchEngine  # noqa: E402
from jarvis.tools.internet import SearchDiagnostics, SearchResult  # noqa: E402
from jarvis.utils.text import normalize_text  # noqa: E402


TEXT_CASES = (
    ("متن: علی نیامد چون بیمار بود. چرا علی نیامد؟", ("بیمار",)),
    ("متن: جلسه لغو شد زیرا برق قطع شد. چرا جلسه لغو شد؟", ("برق",)),
    ("متن: سارا با قطار رفت چون جاده بسته بود. چرا با قطار رفت؟", ("جاده",)),
    ("متن: فایل در پوشه Downloads است. فایل کجاست؟", ("downloads",)),
    ("Text: The build failed because a dependency was missing. Why did it fail?", ("dependency",)),
    ("Text: The meeting begins on Tuesday. When does the meeting begin?", ("tuesday",)),
    ("Text: Mina chose SQLite because transactions were required. Why SQLite?", ("transactions",)),
    ("متن: پروژه آلفا از SQLite استفاده می‌کند. پروژه از چه چیزی استفاده می‌کند؟", ("sqlite",)),
)

LOGIC_CASES = (
    ("همه پرنده‌ها بال دارند و گنجشک پرنده است. چه نتیجه‌ای می‌گیری؟", "fa", ("گنجشک", "بال")),
    ("همه فلزها رسانا هستند و مس فلز است. چه نتیجه‌ای می‌گیری؟", "fa", ("مس", "رسانا")),
    ("همه پستانداران خون گرم هستند و نهنگ پستاندار است. چه نتیجه‌ای می‌گیری؟", "fa", ("نهنگ", "خون گرم")),
    ("همه گربه‌ها حیوان هستند و میلو گربه است. چه نتیجه‌ای می‌گیری؟", "fa", ("میلو", "حیوان")),
    ("All birds are animals. A sparrow is a bird. What follows?", "en", ("sparrow", "animal")),
    ("All mammals are warm blooded. A whale is a mammal. What follows?", "en", ("whale", "warm")),
    ("All roses are flowers. All flowers are plants. A damask is a rose. What follows?", "en", ("damask", "plant")),
)

KNOWLEDGE_CASES = (
    ("NAT چیکار می‌کند؟", ("ip", "آدرس")),
    ("فرق روتر و سوییچ چیست؟", ("mac", "ip")),
    ("VPN چیست؟", ("تونل",)),
    ("فایروال چه کاری انجام می‌دهد؟", ("ترافیک",)),
    ("فرق پهنای باند و latency چیست؟", ("ظرفیت", "تاخیر")),
    ("فرق cache و cookie چیست؟", ("مرورگر",)),
    ("SQL و NoSQL چه فرقی دارند؟", ("جدول", "nosql")),
    ("GitHub همان Git است؟", ("github", "git")),
    ("فرق frontend و backend چیست؟", ("server", "کاربر")),
    ("stack و heap چه هستند؟", ("stack", "heap")),
    ("ماشین مجازی با container چه فرقی دارد؟", ("kernel", "سیستم")),
    ("Big O چیست؟", ("رشد",)),
    ("recursion چیست؟", ("شرط", "تابع")),
    ("فرق authentication و authorization چیست؟", ("هویت", "اجازه")),
    ("فرق رمزنگاری متقارن و نامتقارن چیست؟", ("کلید",)),
    ("فیشینگ چیست؟", ("جعلی",)),
    ("قانون بکاپ 3-2-1 چیست؟", ("سه نسخه",)),
    ("همبستگی یعنی علیت؟", ("علیت", "همبستگی")),
    ("فرق جرم و وزن چیست؟", ("گرانش",)),
    ("فرق climate و weather چیست؟", ("بلندمدت", "کوتاه")),
    ("خورشیدگرفتگی با ماه‌گرفتگی چه فرقی دارد؟", ("سایه",)),
    ("WiFi و Bluetooth چه تفاوتی دارند؟", ("برد",)),
    ("Unicode و UTF-8 یکی هستند؟", ("code point", "byte")),
    ("کدهای 4xx و 5xx HTTP چه هستند؟", ("client", "server")),
)

NEGATIVE_CASES = (
    "متن: هوا بارانی است چون بخار آب متراکم شد. قیمت طلا چقدر است؟",
    "متن: علی در تهران زندگی می‌کند. نسخه بعدی پایتون چه روزی منتشر می‌شود؟",
    "Text: The file is blue. What is today's Bitcoin price?",
    "Text: Sara owns a bicycle. Who won the match last night?",
    "متن: پروژه از SQLite استفاده می‌کند. نتیجه قطعی انتخابات آینده چیست؟",
    "متن: کتاب روی میز است. رمز حساب کاربر چیست؟",
)


class _FixtureInternet:
    def search_detailed(self, query: str) -> SearchDiagnostics:
        return SearchDiagnostics(
            (
                SearchResult(
                    f"{query} official reference", "https://example.gov/reference",
                    f"The official reference documents {query} with directly supported information.",
                    "fixture_official", f"The official reference documents {query} with directly supported information.",
                ),
                SearchResult(
                    f"{query} independent reference", "https://example.edu/research",
                    f"Independent research also documents {query} with supported information.",
                    "fixture_academic", f"Independent research also documents {query} with supported information.",
                ),
            ),
            ("fixture_official", "fixture_academic"), (),
        )

    def fetch_text(self, url: str) -> Any:
        raise AssertionError("Fixture results already contain text")


def _contains(text: str, expected: tuple[str, ...]) -> bool:
    folded = normalize_text(text)
    return all(normalize_text(value) in folded for value in expected)


def evaluate() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reasoner = TextReasoner()
    for query, expected in TEXT_CASES:
        outcome = reasoner.answer_question(query)
        passed = outcome is not None and _contains(outcome.text, expected)
        rows.append({"category": "text_qa", "input": query, "passed": passed})
    for query, language, expected in LOGIC_CASES:
        outcome = reasoner.logical_inference(query, language)
        passed = outcome is not None and _contains(outcome.text, expected)
        rows.append({"category": "formal_reasoning", "input": query, "passed": passed})
    for query in NEGATIVE_CASES:
        passed = reasoner.answer_question(query) is None
        rows.append({"category": "unsupported_claim_guard", "input": query, "passed": passed})

    previous = os.environ.get("JARVIS_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="jarvis-grounding-eval-") as temporary:
        os.environ["JARVIS_DATA_DIR"] = temporary
        config = load_config(ROOT)
        store = KnowledgeStore(
            config.paths.knowledge_database,
            config.paths.knowledge_seed,
            config.knowledge.seed_version,
            config.knowledge.max_documents,
        )
        try:
            for query, expected in KNOWLEDGE_CASES:
                hits = store.search(query, 1)
                passed = bool(hits) and hits[0].score >= config.knowledge.minimum_score
                passed = passed and _contains(hits[0].content, expected)
                rows.append({"category": "offline_knowledge", "input": query, "passed": passed})
        finally:
            store.close()
    if previous is None:
        os.environ.pop("JARVIS_DATA_DIR", None)
    else:
        os.environ["JARVIS_DATA_DIR"] = previous

    search = SearchEngine(_FixtureInternet(), 2)  # type: ignore[arg-type]
    for query in (
        "Python stable release", "TCP congestion control", "DNS resolution",
        "SQLite transactions", "Rayleigh scattering",
    ):
        report = search.search(query)
        passed = bool(report.summary) and len(report.evidence) == 2 and "[" in report.summary
        rows.append({"category": "search_grounding", "input": query, "passed": passed})

    counts = Counter(row["category"] for row in rows)
    passed_counts = Counter(row["category"] for row in rows if row["passed"])
    failed = [row for row in rows if not row["passed"]]
    return {
        "evaluation": "grounded_intelligence_v071",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_network_executed": False,
        "live_network_note": "External networking is environment-dependent; provider protocol is covered by parser and integration tests.",
        "total_cases": len(rows),
        "passed": len(rows) - len(failed),
        "failed": len(failed),
        "accuracy": round((len(rows) - len(failed)) / max(1, len(rows)), 6),
        "categories": {
            category: {
                "cases": counts[category],
                "passed": passed_counts[category],
                "accuracy": round(passed_counts[category] / counts[category], 6),
            }
            for category in sorted(counts)
        },
        "failures": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate JARVIS v0.7.1 grounded reasoning")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "models" / "grounding_evaluation_v071.json",
    )
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
