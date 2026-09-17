from __future__ import annotations

import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

from jarvis.agent.permissions import PermissionLayer
from jarvis.config import FileLimits
from jarvis.neural.conversation import NeuralConversationEngine
from jarvis.neural.transformer import JarvisTransformer
from jarvis.tools.files import FileManager, FileToolError
from jarvis.tools.registry import ToolError, ToolRegistry, ToolSpec
from jarvis.tools.terminal import CommandPolicy
from tests.helpers import ROOT, TemporaryRuntime


class HardeningV08Tests(unittest.TestCase):
    def test_git_alias_and_config_override_are_blocked(self) -> None:
        self.assertEqual(CommandPolicy.assess("git --version").level, "safe")
        self.assertEqual(CommandPolicy.assess("git status").level, "dangerous")
        for command in (
            "git -c alias.pwn=!calc pwn",
            "git --config-env=alias.pwn=VALUE pwn",
            "git alias.pwn something",
        ):
            with self.subTest(command=command):
                self.assertEqual(CommandPolicy.assess(command).level, "blocked")

    def test_declarative_risk_floor_cannot_be_bypassed(self) -> None:
        registry = ToolRegistry(PermissionLayer())
        registry.register(ToolSpec("mutate", "test", "information", lambda _args: True, risk="confirm"))
        with self.assertRaises(ToolError):
            registry.invoke("mutate")
        self.assertTrue(registry.invoke("mutate", confirmed=True))
        self.assertEqual(registry.spec("mutate").schema()["permission_level"], "L2")  # type: ignore[union-attr]

    def test_runtime_categories_protect_ui_browser_clipboard_and_processes(self) -> None:
        with TemporaryRuntime() as runtime:
            levels = {
                name: runtime.tools.spec(name).schema()["permission_level"]  # type: ignore[union-attr]
                for name in ("type_text", "browser_fill", "clipboard_read", "clipboard_clear", "close_app")
            }
        self.assertEqual(levels["type_text"], "L2")
        self.assertEqual(levels["browser_fill"], "L2")
        self.assertEqual(levels["clipboard_read"], "L2")
        self.assertEqual(levels["clipboard_clear"], "L3")
        self.assertEqual(levels["close_app"], "L3")

    def test_file_mutations_never_silently_overwrite(self) -> None:
        limits = FileLimits(25_000_000, 65_536, 100, 1_048_576, 2_097_152)
        manager = FileManager(limits)
        with tempfile.TemporaryDirectory(prefix="jarvis-v08-files-") as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            destination = root / "destination.txt"
            source.write_text("source", encoding="utf-8")
            destination.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileToolError):
                manager.write_text(destination, "replace")
            with self.assertRaises(FileToolError):
                manager.copy_file(source, destination)
            with self.assertRaises(FileToolError):
                manager.move_file(source, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "keep")
            self.assertTrue(source.exists())

    def test_append_can_be_restored_to_previous_size(self) -> None:
        manager = FileManager(FileLimits(25_000_000, 65_536, 100, 1_048_576, 2_097_152))
        with tempfile.TemporaryDirectory(prefix="jarvis-v08-undo-") as temporary:
            path = Path(temporary) / "notes.txt"
            path.write_text("before", encoding="utf-8")
            result = manager.append_text(path, " after")
            self.assertEqual(result.previous_size, 6)
            manager.truncate_file(path, result.previous_size or 0)
            self.assertEqual(path.read_text(encoding="utf-8"), "before")

    def test_docx_rejects_oversized_compressed_xml_member(self) -> None:
        manager = FileManager(FileLimits(25_000_000, 65_536, 100, 128_000, 256_000))
        with tempfile.TemporaryDirectory(prefix="jarvis-v08-docx-") as temporary:
            path = Path(temporary) / "bomb.docx"
            xml = b"<document>" + b"x" * 400_000 + b"</document>"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("word/document.xml", xml)
            with self.assertRaisesRegex(FileToolError, "budget|compression"):
                manager.inspect(path)

    def test_memory_disabled_does_not_persist_message_content(self) -> None:
        marker = "PRIVATE-V08-MARKER-94f3"
        with TemporaryRuntime() as runtime:
            runtime.agent.update_setting("memory_enabled", False)
            runtime.agent.respond(marker)
            logs = runtime.config.paths.logs_dir
            persisted = "".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in logs.glob("*.jsonl") if path.is_file()
            )
            self.assertNotIn(marker, persisted)

    def test_dialogue_adapter_handles_unseen_paraphrases(self) -> None:
        engine = NeuralConversationEngine(
            ROOT / "models" / "jarvis_nano_v08.npz",
            ROOT / "models" / "jarvis_tokenizer_v003.json",
        )
        self.assertFalse(engine.loaded)
        for prompt in (
            "حوصلم خیلی سر رفته، پیشنهادی داری؟",
            "بیا چند دقیقه با هم گپ بزنیم",
            "Could you help me get started?",
            "Tell me one useful thought",
        ):
            with self.subTest(prompt=prompt):
                candidate = engine.generate(prompt, seed=42, sampling_profile="Precise")
                self.assertTrue(candidate.accepted)
                self.assertEqual(candidate.reason, "dialogue_adapter")
                self.assertNotIn("<tool", candidate.text.casefold())
        self.assertFalse(engine.loaded, "adapter replies should not load the full model")

    def test_decoder_honors_immediate_cancellation(self) -> None:
        model = JarvisTransformer.load(ROOT / "models" / "jarvis_nano_v08.npz")
        cancelled = threading.Event()
        cancelled.set()
        generated = model.sample([1, 5, 6], should_cancel=cancelled.is_set)
        self.assertEqual(generated, [])

    def test_smalltalk_routes_to_local_adapter_without_web_search(self) -> None:
        prompts = (
            "حوصلم خیلی سر رفته، پیشنهادی داری؟",
            "الان می‌تونی کنارم باشی؟",
            "I'm feeling a little stuck today",
            "Could you help me get started?",
            "What should we talk about?",
        )
        with TemporaryRuntime() as runtime:
            for prompt in prompts:
                with self.subTest(prompt=prompt):
                    self.assertEqual(runtime.agent.router.route(prompt).intent, "smalltalk")
                    reply = runtime.agent.respond(prompt)
                    self.assertEqual(reply.intent, "neural_conversation")
                    self.assertEqual(reply.data.get("sampling_profile"), "Balanced/Adapter")

    def test_smalltalk_prefix_cannot_hide_a_dangerous_request(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("Could you help me delete every file?")
        self.assertEqual(route.intent, "dangerous_request")
        self.assertTrue(route.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
