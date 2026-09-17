from __future__ import annotations

import unittest
from unittest import mock

from jarvis.agent.challenge_reasoner import ChallengeReasoner
from jarvis.agent.dialogue_subject import DialogueSubjectResolver
from tests.helpers import TemporaryRuntime


class ChallengeReasoningV094Tests(unittest.TestCase):
    LOGIC_CASES = (
        (
            "سه جعبه داریم با برچسب‌های سیب، پرتقال و مخلوط؛ هر سه برچسب اشتباه‌اند. "
            "فقط با بیرون‌آوردن یک میوه، چطور برچسب همه را درست می‌کنی؟",
            ("برچسب «مخلوط»", "جعبهٔ باقی‌مانده"),
        ),
        (
            "دو طناب داریم که هرکدام دقیقاً ۶۰ دقیقه می‌سوزند، اما سرعت سوختنشان یکنواخت نیست. "
            "چطور دقیقاً ۴۵ دقیقه را اندازه می‌گیری؟",
            ("۳۰", "۱۵", "۴۵"),
        ),
        (
            "همهٔ زِپ‌ها لور هستند. بعضی لورها آبی‌اند. آیا حتماً بعضی زپ‌ها آبی‌اند؟ با دلیل جواب بده.",
            ("نه", "مثال نقض"),
        ),
        (
            "اگر دیروز، فردای شنبه بود، امروز چه روزی است؟ اول فرض جمله را دقیق تفسیر کن.",
            ("دیروز = فردای شنبه", "دوشنبه"),
        ),
        (
            "در اتاقی سه کلید و در اتاق بستهٔ دیگری سه لامپ وجود دارد. فقط یک بار اجازه داری وارد "
            "اتاق لامپ‌ها شوی؛ چطور هر کلید را شناسایی می‌کنی؟",
            ("خاموشِ گرم", "خاموشِ سرد"),
        ),
        (
            "دنبالهٔ ۱، ۲، ۴، ۸ را ادامه بده؛ اما قبلش توضیح بده چرا جواب قطعی و یکتا نیست.",
            ("16", "یکتا نیست"),
        ),
    )

    def test_reported_logic_questions_are_solved_locally_without_tools(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("desktop/web tool call was not expected")
        ):
            for question, fragments in self.LOGIC_CASES:
                with self.subTest(question=question):
                    answer = runtime.agent.respond(question)
                    self.assertEqual(answer.intent, "reasoned_answer")
                    self.assertEqual(answer.data["reasoning_source"], "local_challenge_reasoner_v094")
                    for fragment in fragments:
                        self.assertIn(fragment, answer.text)

    def test_quantitative_solvers_generalize_beyond_the_reported_values(self) -> None:
        reasoner = ChallengeReasoner()
        cases = (
            ("زاویه بین عقربه‌ها در ساعت ۶:۳۰ چند درجه است؟", "15 درجه"),
            ("تعداد صفرهای انتهای 25! را پیدا کن.", "6 صفر"),
            (
                "اگر احتمال موفقیت هر آزمایش مستقل ۵۰٪ باشد، احتمال حداقل یک موفقیت در سه آزمایش چقدر است؟",
                "87.5٪",
            ),
            ("عدد بعدی دنباله ۳، ۶، ۱۲، ۲۴ چیست؟", "48"),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                answer = reasoner.solve(question)
                self.assertIsNotNone(answer)
                assert answer is not None
                self.assertIn(expected, answer.text)

    def test_math_prompt_batch_is_isolated_and_all_five_items_are_answered(self) -> None:
        prompt = """ریاضی و دام‌های ذهنی
«یک توپ و چوب روی‌هم ۱۱۰ هزار تومان‌اند. چوب ۱۰۰ هزار تومان از توپ گران‌تر است. قیمت توپ چقدر است؟ مرحله‌ای بررسی کن.»
«زاویهٔ دقیق بین عقربه‌های ساعت در ساعت ۳:۱۵ چند درجه است؟»
«تعداد صفرهای انتهای عدد 100! را بدون محاسبهٔ کامل فاکتوریل پیدا کن.»
«اگر احتمال موفقیت هر آزمایش مستقل ۲۰٪ باشد، احتمال حداقل یک موفقیت در پنج آزمایش چقدر است؟»
«عدد بعدی دنبالهٔ ۲، ۳، ۵، ۹، ۱۷ چیست؟ اگر چند قاعده ممکن است، صریح بگو.»"""
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web search was not expected")
        ):
            answer = runtime.agent.respond(prompt)
        self.assertEqual(answer.data["reasoning_source"], "local_challenge_reasoner_v094")
        for expected in ("5,000 تومان", "7.5 درجه", "24 صفر", "67.232٪", "33"):
            self.assertIn(expected, answer.text)
        self.assertEqual(answer.data["self_checks"][-1], "all_items_locally_solved")

    def test_pasted_memory_script_cannot_inherit_stale_translation_subject(self) -> None:
        prompt = """این پنج پیام را پشت‌سرهم بفرست:
«کد پروژهٔ من آبی-۴۷ است؛ فعلاً فقط یادت نگه دار.»
«تفاوت RAM و حافظهٔ ذخیره‌سازی چیست؟»
«یک مثال ساده هم بزن.»
«حالا توضیح قبلی را در یک جمله خلاصه کن.»
«کد پروژه‌ای که اول گفتم چه بود؟»
آزمون زنجیره‌ای دوم:
«معادل انگلیسی کلمهٔ پنجره چیست؟»
«حالا جمع همان کلمه در انگلیسی چیست؟»
«با همان کلمه یک جمله بساز.»
«جمله را رسمی‌تر کن، ولی معنایش را تغییر نده.»"""
        resolved = DialogueSubjectResolver.resolve(
            prompt,
            {"last_information_subject": "کیر", "last_information_mode": "english_equivalent"},
        )
        self.assertEqual(resolved.query, " ".join(prompt.split()))
        self.assertFalse(resolved.used_context)
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web search was not expected")
        ):
            runtime.agent.respond("مخفف انگلیسی کلمه کیر چیست")
            answer = runtime.agent.respond(prompt)
        self.assertIn("پیام مستقل", answer.text)
        self.assertIn("یکی‌یکی", answer.text)
        self.assertNotIn("penis", answer.text.casefold())
        self.assertEqual(answer.data["self_checks"][-1], "separate_turns_requested")

    def test_challenge_guard_does_not_hijack_real_datetime_or_close_commands(self) -> None:
        with TemporaryRuntime() as runtime:
            date_route = runtime.agent.router.route("امروز چه روزی است؟", {})
            close_route = runtime.agent.router.route("برنامه نوت‌پد را ببند", {})
        self.assertEqual(date_route.intent, "date_time")
        self.assertEqual(close_route.intent, "close_app")
        self.assertNotEqual(date_route.source, "local_challenge_guard_v094")
        self.assertNotEqual(close_route.source, "local_challenge_guard_v094")

    def test_project_code_and_ram_followups_work_as_separate_turns(self) -> None:
        prompts = (
            "کد پروژهٔ من آبی-۴۷ است؛ فعلاً فقط یادت نگه دار.",
            "تفاوت RAM و حافظهٔ ذخیره‌سازی چیست؟",
            "یک مثال ساده هم بزن.",
            "حالا توضیح قبلی را در یک جمله خلاصه کن.",
            "کد پروژه‌ای که اول گفتم چه بود؟",
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("tool call was not expected")
        ):
            answers = [runtime.agent.respond(prompt) for prompt in prompts]
        self.assertEqual(answers[0].intent, "fact_remembered")
        self.assertIn("SSD", answers[1].text)
        self.assertIn("بازی", answers[2].text)
        self.assertEqual(answers[3].text.count("."), 1)
        self.assertIn("آبی-۴۷", answers[4].text)

    def test_window_translation_chain_keeps_word_number_and_sentence(self) -> None:
        prompts = (
            "معادل انگلیسی کلمهٔ پنجره چیست؟",
            "حالا جمع همان کلمه در انگلیسی چیست؟",
            "با همان کلمه یک جمله بساز.",
            "جمله را رسمی‌تر کن، ولی معنایش را تغییر نده.",
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("tool call was not expected")
        ):
            answers = [runtime.agent.respond(prompt) for prompt in prompts]
        self.assertIn("window", answers[0].text)
        self.assertIn("windows", answers[1].text)
        self.assertEqual(answers[2].text, "The window overlooks the garden.")
        self.assertEqual(answers[3].text, "The window provides a view of the garden.")
        self.assertNotIn("penis", " ".join(answer.text.casefold() for answer in answers))


if __name__ == "__main__":
    unittest.main()
