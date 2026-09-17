from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from jarvis.utils.text import normalize_text
from tests.helpers import ROOT, TemporaryRuntime


HOLDOUT_CASES = (
    ("7!", ("5,040",)),
    ("آیا 29 عدد اول است؟", ("عدد اول است",)),
    ("احتمال سه بار شیر آمدن در سه پرتاب سکه سالم چقدر است؟", ("1/8", "12.5")),
    ("17 درصد از 340 چقدر می‌شود؟", ("57.8",)),
    ("ب.م.م 84 و 126 چیست؟", ("42",)),
    ("ک.م.م 18 و 24 چیست؟", ("72",)),
    ("از 12 عضو چند زیرمجموعه 3 عضوی می‌توان انتخاب کرد؟", ("220",)),
    ("How many ordered selections of 3 items from 10 distinct items?", ("720",)),
    ("C(15,2) را حساب کن", ("105",)),
    ("P(12,4) را حساب کن", ("11,880",)),
    ("میانگین اعداد 12، 18، 24 و 30 چیست؟", ("21",)),
    ("نمره 70 با وزن 2 و نمره 90 با وزن 3 داریم. میانگین وزنی؟", ("82",)),
    ("عدد 90 را به نسبت 2:3 تقسیم کن", ("36", "54")),
    ("نسبت 84 به 126 را ساده کن", ("2:3",)),
    ("Convert 2.75 km to meters", ("2,750",)),
    ("2 کیلوگرم را به گرم تبدیل کن", ("2,000",)),
    ("دمای 25 درجه سانتی‌گراد را به فارنهایت تبدیل کن", ("77",)),
    ("32 درجه فارنهایت را به سانتی گراد تبدیل کن", ("0",)),
    ("دنباله حسابی از 5 با اختلاف 4 داریم. جمله 9 چیست؟", ("37",)),
    ("Geometric sequence starts at 3 with ratio 2. Find term 7.", ("192",)),
    ("دنباله هندسی از 4 با نسبت 3 داریم. جمله 6 چیست؟", ("972",)),
    ("خودرویی با سرعت 65 کیلومتر بر ساعت، 4 ساعت حرکت می‌کند. مسافت چقدر است؟", ("260",)),
    ("1007 را بر 13 تقسیم کن و خارج قسمت و باقیمانده را بگو", ("77", "6")),
    ("عدد 1047 زوج است یا فرد؟", ("فرد",)),
    ("حل کن: 3x + 5 = 29", ("x = 8",)),
    ("سه‌شنبه + 10 روز چه روزی است؟", ("جمعه",)),
    ("به انگلیسی ترجمه کن: من امروز خسته‌ام", ("I am tired today.",)),
    ("Translate to Persian: I need help", ("کمک",)),
    ("یک تابع Python بنویس که اعداد زوج یک لیست را فیلتر کند", ("def filter_even", "% 2 == 0")),
    ("Write Python code that checks whether a number is prime", ("def is_prime",)),
)


class RuntimeIntelligenceV14Tests(unittest.TestCase):
    def test_v14_holdout_prompts_are_not_training_examples(self) -> None:
        paths = (
            ROOT / "datasets" / "splits" / "dataset_v006_train.jsonl",
            ROOT / "datasets" / "splits" / "dataset_v006_validation.jsonl",
            ROOT / "datasets" / "splits" / "dataset_v006_test.jsonl",
        )
        training_inputs: set[str] = set()
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                training_inputs.add(normalize_text(str(row.get("input", ""))).casefold())
        for prompt, _ in HOLDOUT_CASES:
            with self.subTest(prompt=prompt):
                self.assertNotIn(normalize_text(prompt).casefold(), training_inputs)

    def test_v14_end_to_end_reasoning_language_and_code_holdout(self) -> None:
        # One continuous session is intentional: these prompts must keep routing
        # correctly even after unrelated earlier turns have populated context.
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                for prompt, fragments in HOLDOUT_CASES:
                    with self.subTest(prompt=prompt):
                        answer = runtime.agent.respond(prompt)
                        lowered = answer.text.casefold()
                        for fragment in fragments:
                            self.assertIn(fragment.casefold(), lowered)
                        self.assertNotIn(answer.intent, {"unknown_information", "context_clarification", "tool_search_result"})

    def test_v14_dataset_quality_and_diversity_gates(self) -> None:
        generalization = json.loads((ROOT / "datasets" / "generalization_100_v14_manifest.json").read_text(encoding="utf-8"))
        reasoning = json.loads((ROOT / "datasets" / "reasoning_100_v14_manifest.json").read_text(encoding="utf-8"))
        merged = json.loads((ROOT / "datasets" / "manifest_v006.json").read_text(encoding="utf-8"))

        self.assertEqual(generalization["source_dataset_count"], 100)
        self.assertEqual(reasoning["source_dataset_count"], 100)
        self.assertEqual(generalization["examples"], 2000)
        self.assertEqual(reasoning["examples"], 3000)
        self.assertGreaterEqual(generalization["unique_output_ratio"], 0.995)
        self.assertGreaterEqual(reasoning["unique_output_ratio"], 0.98)
        self.assertEqual(generalization["unique_inputs"], generalization["examples"])
        self.assertEqual(reasoning["unique_inputs"], reasoning["examples"])
        self.assertEqual(merged["cross_split_concept_leakage"], 0)
        self.assertTrue(merged["removed_v13_template_packs"])
        self.assertEqual(merged["generalization_source_dataset_count"], 100)
        self.assertEqual(merged["reasoning_source_dataset_count"], 100)

        for folder in (ROOT / "datasets" / "generalization_v14", ROOT / "datasets" / "reasoning_v14"):
            for path in folder.glob("*.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertNotRegex(str(row["input"]), r"^\s*(?:تمرین|exercise)\s*\d+[-:]", msg=str(path))


if __name__ == "__main__":
    unittest.main()
