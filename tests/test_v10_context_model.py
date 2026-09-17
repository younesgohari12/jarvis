from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from jarvis.agent.followup_model import FollowupIntentModel
from tests.helpers import TemporaryRuntime


ROOT = Path(__file__).resolve().parents[1]


class ContextModelV10Tests(unittest.TestCase):
    def test_trained_followup_model_passed_held_out_quality_gate(self) -> None:
        model = FollowupIntentModel(ROOT / "models" / "dialogue_followup_v10.json")
        metrics = json.loads(
            (ROOT / "models" / "dialogue_followup_metrics_v10.json").read_text(encoding="utf-8")
        )
        self.assertTrue(model.ready)
        self.assertGreaterEqual(model.training_examples, 94)
        self.assertIsNone(metrics["pretrained_source"])
        self.assertTrue(metrics["accepted_for_release"])
        self.assertGreaterEqual(metrics["classification_accuracy"], 0.875)
        self.assertEqual(metrics["held_out_examples"], 8)

    def test_reported_chain_uses_memory_and_never_searches(self) -> None:
        prompts = (
            "رمز پروژهٔ جدید من نقره‌ای-۹۳ است؛ به خاطر بسپار.",
            "تفاوت HDD و SSD چیست؟",
            "توضیح را در یک جمله خلاصه کن.",
            "رمز پروژهٔ جدید چه بود؟",
            "معادل انگلیسی کودک چیست؟",
            "جمع آن در انگلیسی چیست؟",
            "با همان کلمه یک جمله بساز.",
            "همان جمله را رسمی‌تر کن.",
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            answers = [runtime.agent.respond(prompt) for prompt in prompts]
        self.assertEqual(answers[0].intent, "fact_remembered")
        self.assertIn("SSD", answers[1].text)
        self.assertEqual(answers[2].intent, "contextual_summary")
        self.assertIn("HDD", answers[2].text)
        self.assertIn("نقره‌ای-۹۳", answers[3].text)
        self.assertIn("child", answers[4].text)
        self.assertIn("children", answers[5].text)
        self.assertEqual(answers[6].text, "The child completed the task.")
        self.assertEqual(answers[7].text, "The child successfully completed the assigned task.")

    def test_missing_translation_context_clarifies_instead_of_using_stale_word(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            runtime.agent.respond("معادل انگلیسی کلمه مرد چیست؟")
            runtime.agent.respond("تفاوت HDD و SSD چیست؟")
            answer = runtime.agent.respond("جمع آن در انگلیسی چیست؟")
        self.assertEqual(answer.intent, "context_clarification")
        self.assertIn("مرجع واژه", answer.text)
        self.assertNotIn("penis", answer.text.casefold())
        self.assertNotIn("men", answer.text.casefold())

    def test_independent_translation_replaces_old_language_context(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            runtime.agent.respond("معادل انگلیسی کلمه مرد چیست؟")
            child = runtime.agent.respond("معادل انگلیسی کودک چیست؟")
            plural = runtime.agent.respond("جمع آن در انگلیسی چیست؟")
        self.assertIn("child", child.text)
        self.assertIn("children", plural.text)
        self.assertNotIn("men", plural.text.casefold())

    def test_generic_user_facts_store_and_recall_different_labels(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            saved_order = runtime.agent.respond("کد سفارش من ZX-51 است؛ یادت بماند")
            saved_server = runtime.agent.respond("شناسه سرور من dev-8 است؛ ثبت کن")
            order = runtime.agent.respond("کد سفارش چه بود؟")
            server = runtime.agent.respond("شناسه سرور قبلی چی بود؟")
        self.assertEqual(saved_order.intent, "fact_remembered")
        self.assertEqual(saved_server.intent, "fact_remembered")
        self.assertIn("ZX-51", order.text)
        self.assertIn("dev-8", server.text)

    def test_incomplete_summary_without_recent_answer_stays_local(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            answer = runtime.agent.respond("توضیح قبلی را در یک جمله خلاصه کن")
        self.assertEqual(answer.intent, "context_clarification")
        self.assertIn("مشخص نیست", answer.text)
        self.assertNotIn("منابع", answer.text)


if __name__ == "__main__":
    unittest.main()
