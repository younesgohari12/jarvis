from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from jarvis.gui.app import JarvisGUI
from tests.helpers import TemporaryRuntime


class ReportedRegressionsV095Tests(unittest.TestCase):
    def test_all_none_syllogism_and_compound_weekday_are_solved_locally(self) -> None:
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            logic = runtime.agent.respond(
                "اگر همهٔ نِپ‌ها رال باشند و هیچ رالی سبز نباشد، آیا ممکن است یک نِپ سبز باشد؟ دلیل بیاور."
            )
            weekday = runtime.agent.respond(
                "اگر دیروز، فردای چهارشنبه بود، امروز چه روزی است؟ جمله را دقیق تفسیر کن."
            )
        self.assertEqual(logic.data["reasoning_source"], "local_challenge_reasoner_v094")
        self.assertIn("ممکن نیست", logic.text)
        self.assertIn("تناقض", logic.text)
        self.assertIn("فردای چهارشنبه", weekday.text)
        self.assertIn("جمعه", weekday.text)
        self.assertNotIn("شنبه». فردای شنبه", weekday.text)

    def test_secret_cpu_example_and_recall_chain_never_searches(self) -> None:
        prompts = (
            "رمز پروژهٔ من سبز-۸۲ است؛ فقط یادت نگه دار.",
            "تفاوت CPU و GPU چیست؟",
            "یک مثال کوتاه بزن.",
            "رمز پروژه‌ای که گفتم چه بود؟",
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            answers = [runtime.agent.respond(prompt) for prompt in prompts]
        self.assertEqual(answers[0].intent, "fact_remembered")
        self.assertIn("CPU", answers[1].text)
        self.assertIn("پیکسل", answers[2].text)
        self.assertIn("سبز-۸۲", answers[3].text)

    def test_child_translation_plural_sentence_and_formality_keep_context(self) -> None:
        prompts = (
            "معادل انگلیسی کودک چیست؟",
            "جمع آن در انگلیسی چیست؟",
            "با همان کلمه یک جمله بساز.",
            "همان جمله را رسمی‌تر کن.",
        )
        with TemporaryRuntime() as runtime, mock.patch.object(
            runtime.tools, "invoke", side_effect=AssertionError("web/tool call was not expected")
        ):
            answers = [runtime.agent.respond(prompt) for prompt in prompts]
        self.assertIn("child", answers[0].text)
        self.assertIn("children", answers[1].text)
        self.assertEqual(answers[2].text, "The child completed the task.")
        self.assertEqual(answers[3].text, "The child successfully completed the assigned task.")
        self.assertNotIn("penis", " ".join(answer.text.casefold() for answer in answers))

    def test_input_supports_virtual_paste_shift_insert_right_click_and_layout_fallback(self) -> None:
        class FakeText:
            def __init__(self) -> None:
                self.bindings: dict[str, object] = {}

            def bind(self, sequence: str, callback: object) -> None:
                self.bindings[sequence] = callback

        gui = object.__new__(JarvisGUI)
        gui.message_input = FakeText()  # type: ignore[assignment]
        gui._bind_input_clipboard_shortcuts()
        expected = {
            "<<Paste>>", "<Control-v>", "<Control-V>", "<Shift-Insert>",
            "<Control-KeyPress>", "<Button-3>",
        }
        self.assertTrue(expected.issubset(gui.message_input.bindings))

        gui._paste_input_clipboard = mock.Mock(return_value="break")  # type: ignore[method-assign]
        event = SimpleNamespace(keysym="Arabic_veh", keycode=86)
        self.assertEqual(gui._clipboard_layout_keypress(event), "break")  # type: ignore[arg-type]
        gui._paste_input_clipboard.assert_called_once_with(event)

    def test_busy_composer_disables_input_and_prevents_a_second_queued_message(self) -> None:
        class Widget:
            def __init__(self) -> None:
                self.state = "normal"
                self.focused = False

            def configure(self, **kwargs: object) -> None:
                self.state = str(kwargs.get("state", self.state))

            def focus_set(self) -> None:
                self.focused = True

        gui = object.__new__(JarvisGUI)
        gui.send_button = Widget()  # type: ignore[assignment]
        gui.message_input = Widget()  # type: ignore[assignment]
        gui._set_chat_busy(True)
        self.assertTrue(gui._chat_busy)
        self.assertEqual(gui.send_button.state, "disabled")
        self.assertEqual(gui.message_input.state, "disabled")
        gui._set_chat_busy(False)
        self.assertFalse(gui._chat_busy)
        self.assertEqual(gui.message_input.state, "normal")
        self.assertTrue(gui.message_input.focused)


if __name__ == "__main__":
    unittest.main()
