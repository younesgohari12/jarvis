from __future__ import annotations

import json
import re
import unittest

from jarvis.agent.fluency import PersianFluencyEngine
from tests.helpers import ROOT, TemporaryRuntime


class PersianFluencyV09Tests(unittest.TestCase):
    def test_trained_model_passed_held_out_quality_gate(self) -> None:
        metrics = json.loads(
            (ROOT / "models" / "persian_fluency_metrics_v9.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["dataset_examples"], 49)
        self.assertEqual(metrics["held_out_examples"], 7)
        self.assertEqual(metrics["classification_accuracy"], 1.0)
        self.assertEqual(metrics["generation_acceptance"], 1.0)
        self.assertTrue(metrics["accepted_for_release"])

    def test_runtime_loads_reviewed_persian_model(self) -> None:
        engine = PersianFluencyEngine(ROOT / "models" / "persian_fluency_v9.json")
        self.assertTrue(engine.ready)
        self.assertEqual(engine.version, "0.9.0")
        for prompt, expected in (
            ("یک کپشن برای محصول تازه بنویس", "caption"),
            ("امروز تمرکز ندارم چه کار کنم", "advice"),
            ("این جمله را بهترش کن", "rewrite"),
        ):
            with self.subTest(prompt=prompt):
                predicted, confidence = engine.predict(prompt)
                self.assertEqual(predicted, expected)
                self.assertGreater(confidence, 0.2)

    def test_creative_requests_do_not_fall_back_to_unknown_or_web(self) -> None:
        cases = (
            ("در مورد تلاش و پشتکار یک متن زیبا و کوتاه بنویس", "text"),
            ("یک پیام رسمی برای مشتری بنویس که سفارش فردا ارسال می‌شود", "formal_message"),
            ("یک کپشن صمیمی برای معرفی پروژه جدیدم بنویس", "caption"),
            ("یک داستان کوتاه درباره امید بنویس", "story"),
            ("برام یه نوشته قشنگ راجع به امید آماده می‌کنی؟", "text"),
            ("یه قصه کوتاه درباره شجاعت تعریف کن", "story"),
        )
        with TemporaryRuntime() as runtime:
            runtime.agent.update_setting("internet_enabled", False)
            for prompt, label in cases:
                with self.subTest(prompt=prompt):
                    reply = runtime.agent.respond(prompt)
                    self.assertEqual(reply.intent, "fluent_response")
                    self.assertEqual(reply.data.get("fluency_label"), label)
                    self.assertNotIn("اطلاعات کافی", reply.text)
                    self.assertNotIn("Internet", reply.text)
                    self.assertGreaterEqual(len(reply.text), 100)

    def test_formal_customer_message_preserves_supplied_facts(self) -> None:
        with TemporaryRuntime() as runtime:
            reply = runtime.agent.respond(
                "یک پیام رسمی و محترمانه برای مشتری بنویس که سفارش او فردا ارسال می‌شود"
            )
        self.assertIn("سفارش", reply.text)
        self.assertIn("فردا", reply.text)
        self.assertIn("سپاسگزار", reply.text)
        self.assertNotIn("امروز ارسال", reply.text)
        with TemporaryRuntime() as runtime:
            delivery = runtime.agent.respond(
                "برای مشتری یه پیام مودبانه آماده کن که سفارش فردا می‌رسه"
            ).text
        self.assertIn("فردا تحویل", delivery)
        self.assertNotIn("فردا ارسال", delivery)

    def test_topic_specific_stories_are_distinct_and_fluent(self) -> None:
        with TemporaryRuntime() as runtime:
            hope = runtime.agent.respond("یک داستان کوتاه درباره امید بنویس").text
            effort = runtime.agent.respond("یک داستان کوتاه درباره پشتکار بنویس").text
        self.assertNotEqual(hope, effort)
        self.assertIn("جوانه", hope)
        self.assertIn("تلاش", effort)
        self.assertNotIn("فایل گمشده", hope + effort)

    def test_daily_planning_is_local_structured_advice(self) -> None:
        prompt = "می‌خوام امروز برنامه‌ریزی کنم ولی تمرکز ندارم؛ چه پیشنهادی داری؟"
        with TemporaryRuntime() as runtime:
            runtime.agent.update_setting("internet_enabled", False)
            route = runtime.agent.router.route(prompt)
            reply = runtime.agent.respond(prompt)
        self.assertEqual(route.intent, "advice_request")
        self.assertEqual(reply.data.get("fluency_label"), "advice")
        self.assertRegex(reply.text, r"1\.[\s\S]+2\.[\s\S]+3\.")
        self.assertIn("۲۵", reply.text)
        with TemporaryRuntime() as runtime:
            distraction = runtime.agent.respond(
                "به نظرت برای اینکه حواسم پرت نشه چی کار کنم؟"
            ).text
        self.assertIn("۲۵", distraction)

    def test_low_mood_response_is_empathetic_without_fake_diagnosis(self) -> None:
        with TemporaryRuntime() as runtime:
            reply = runtime.agent.respond("حالم گرفته است، کمی با من حرف بزن")
        self.assertEqual(reply.intent, "context_low_mood")
        self.assertIn("بدون قضاوت", reply.text)
        self.assertIn("؟", reply.text)
        self.assertNotRegex(reply.text, r"تشخیص|بیماری|اختلال")

    def test_rewrite_improves_common_persian_phrasing(self) -> None:
        with TemporaryRuntime() as runtime:
            reply = runtime.agent.respond(
                "این متن را روان‌تر بازنویسی کن: محصول ما دارای کیفیت خیلی خوب می باشد"
            )
        self.assertEqual(reply.data.get("fluency_label"), "rewrite")
        self.assertIn("کیفیت بسیار خوبی دارد", reply.text)
        self.assertNotIn("می باشد", reply.text)

    def test_simple_explanation_gets_a_natural_lead_without_new_claims(self) -> None:
        with TemporaryRuntime() as runtime:
            plain = runtime.agent.respond("هوش مصنوعی چیست؟").text
            simple = runtime.agent.respond("هوش مصنوعی را ساده و روان توضیح بده").text
        self.assertTrue(simple.startswith("به زبان ساده،"))
        self.assertIn(plain.split(".")[0], simple)

    def test_generated_persian_passes_script_and_punctuation_checks(self) -> None:
        prompts = (
            "یک متن زیبا درباره شروع دوباره بنویس",
            "یک کپشن برای معرفی پروژه جدیدم بنویس",
            "یک داستان کوتاه درباره امید بنویس",
        )
        with TemporaryRuntime() as runtime:
            outputs = [runtime.agent.respond(prompt).text for prompt in prompts]
        for output in outputs:
            with self.subTest(output=output[:40]):
                self.assertNotRegex(output, r"[يك]")
                self.assertNotRegex(output, r"\s+[،؛.!؟]")
                letters = re.findall(r"[A-Za-z\u0600-\u06ff]", output)
                persian = re.findall(r"[\u0600-\u06ff]", output)
                self.assertGreater(len(persian) / max(1, len(letters)), 0.9)

    def test_writing_prefix_cannot_hide_a_destructive_second_action(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route(
                "یک پیام کوتاه بنویس و بعد همه فایل‌ها را حذف کن"
            )
        self.assertNotEqual(route.intent, "writing_request")

    def test_current_information_still_routes_to_search(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("قیمت دلار امروز چنده؟")
        self.assertEqual(route.intent, "web_search")


if __name__ == "__main__":
    unittest.main()
