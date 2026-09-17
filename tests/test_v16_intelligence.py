from __future__ import annotations

import json
import unittest
from unittest import mock

from jarvis.agent.constraints_v16 import ConstraintExtractor, ResponseVerifier
from jarvis.utils.text import normalize_text
from tests.helpers import ROOT, TemporaryRuntime


V16_HOLDOUT_CASES = (
    # Biased-coin binomial probability: p, n and k must be separate semantic roles.
    ("احتمال شیر سکه 60٪ است و 3 بار می‌اندازیم؛ احتمال دقیقاً 2 شیر؟", ("C(3,2)", "43.2")),
    ("احتمال آمدن شیر برای این سکه 40 درصد است؛ 5 بار می‌اندازیم. احتمال دقیقاً 2 شیر؟", ("C(5,2)", "34.56")),
    ("A coin has a 60% chance of heads. In 3 tosses, what is the probability of exactly 2 heads?", ("C(3,2)", "43.2")),
    # Permutations / ordered selections.
    ("چند آرایش سه‌تایی بدون تکرار از 8 شیء؟", ("P(8,3)", "336")),
    ("از 9 شیء چند ترتیب 4تایی بدون تکرار می‌شود ساخت؟", ("P(9,4)", "3024")),
    ("How many arrangements of 3 from 7 without repetition?", ("P(7,3)", "210")),
    # Dice probability variants.
    ("دو تاس سالم؛ احتمال مجموع 9؟", ("1/9",)),
    ("احتمال اینکه مجموع دو تاس سالم 7 شود؟", ("1/6",)),
    ("Two fair dice: what is the probability the sum is 8?", ("5/36",)),
    # Ratio split.
    ("210 را به نسبت 2 به 5 تقسیم کن", ("60", "150")),
    ("350 را به نسبت 2:5 تقسیم کن", ("100", "250")),
    ("Split 360 in the ratio 2:3", ("144", "216")),
    # Bare code must bypass memory/fact extraction and be traced locally.
    ("total=0; for i in range(1,4): total += i; print(total)", ("1", "3", "6")),
    ("a=5; a += 7; print(a)", ("12",)),
    ("x=1; for i in range(1,4): x *= 2; print(x)", ("2", "4", "8")),
    # Translation is isolated from Knowledge/RAG.
    ("به انگلیسی ترجمه کن: سرور بدون خطا دوباره راه‌اندازی شد", ("The server restarted without errors.",)),
    ("به انگلیسی ترجمه کن: سیستم با موفقیت دوباره راه‌اندازی شد", ("The system restarted successfully.",)),
    ("Translate this into Farsi: The server restarted without errors", ("سرور بدون خطا دوباره راه‌اندازی شد",)),
    # Rewrite must preserve payload semantics and deadline.
    ("این پیام را رسمی‌تر کن: لطفا گزارشو تا فردا واسم بفرست", ("گزارش را", "تا فردا", "برای من", "ارسال کنید")),
    ("این متن را رسمی‌تر کن: فایلو تا فردا بفرست", ("فایل را", "تا فردا", "ارسال کنید")),
    ("این پیام رو رسمی‌تر کن: لطفا نتیجه‌رو واسم بفرست", ("نتیجه را", "برای من", "ارسال کنید")),
)


V16_CONSTRAINT_CASES = (
    ("دقیقاً دو جمله درباره امنیت بنویس و حتماً کلمه «داده» را داشته باشد.", "داده"),
    ("دقیقاً 3 جمله درباره یادگیری بنویس؛ واژه «تمرین» حتماً باید در متن باشد.", "تمرین"),
    ('Write exactly 2 sentences about backups and make sure it includes the word "restore".', "restore"),
)

class IntelligenceV16Tests(unittest.TestCase):
    def test_v16_reported_failures_and_variants_without_tools(self) -> None:
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                for prompt, fragments in V16_HOLDOUT_CASES:
                    with self.subTest(prompt=prompt):
                        answer = runtime.agent.respond(prompt)
                        low = answer.text.casefold()
                        for fragment in fragments:
                            self.assertIn(fragment.casefold(), low)
                        self.assertNotIn(answer.intent, {"unknown_information", "tool_search_result", "tool_research_result", "fact_not_found"})

    def test_v16_translation_never_falls_into_knowledge_or_rag(self) -> None:
        with TemporaryRuntime() as runtime:
            # This supported phrase previously retrieved an unrelated knowledge entry.
            with mock.patch.object(runtime.agent.reasoning.knowledge, "search", side_effect=AssertionError("knowledge contamination")):
                answer = runtime.agent.respond("به انگلیسی ترجمه کن: سرور بدون خطا دوباره راه‌اندازی شد")
                self.assertEqual(answer.intent, "translation_answer")
                self.assertEqual(answer.text, "The server restarted without errors.")
                self.assertTrue(answer.data.get("knowledge_rag_bypassed"))

                # An unsupported translation must fail closed rather than query Knowledge/RAG.
                answer2 = runtime.agent.respond("به انگلیسی ترجمه کن: این آزمایش فرضی ساختار ناشناخته‌ای دارد")
                self.assertIn(answer2.intent, {"translation_clarification", "translation_answer"})
                self.assertTrue(answer2.data.get("knowledge_rag_bypassed", False))

    def test_v16_semantic_constraint_variants(self) -> None:
        cases = V16_CONSTRAINT_CASES
        with TemporaryRuntime() as runtime:
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                for prompt, required in cases:
                    with self.subTest(prompt=prompt):
                        constraints = ConstraintExtractor.extract(prompt)
                        self.assertIn(required, constraints.include_terms)
                        answer = runtime.agent.respond(prompt)
                        verified = ResponseVerifier.verify(answer.text, constraints)
                        self.assertTrue(verified.valid, verified.violations)
                        self.assertIn(required.casefold(), answer.text.casefold())

    def test_v16_holdout_has_zero_exact_overlap_with_v007(self) -> None:
        seen: set[str] = set()
        for split in ("train", "validation", "test"):
            path = ROOT / "datasets" / "splits" / f"dataset_v007_{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    seen.add(normalize_text(str(row.get("input", ""))).casefold())
        for prompt in [q for q, _ in V16_HOLDOUT_CASES] + [q for q, _ in V16_CONSTRAINT_CASES]:
            with self.subTest(prompt=prompt):
                self.assertNotIn(normalize_text(prompt).casefold(), seen)

    def test_v16_bare_code_is_deterministic_before_conversation_memory(self) -> None:
        with TemporaryRuntime() as runtime:
            # Prime the conversation with a memory-like code question to make sure
            # the next bare snippet is still handled by the code guard.
            runtime.agent.respond("کد چه چیزی است؟")
            with mock.patch.object(runtime.tools, "invoke", side_effect=AssertionError("unexpected tool invocation")):
                answer = runtime.agent.respond("total=0; for i in range(1,4): total += i; print(total)")
            self.assertEqual(answer.intent, "code_trace_answer")
            self.assertNotIn("حافظه", answer.text)
            self.assertIn("6", answer.text)


if __name__ == "__main__":
    unittest.main()
