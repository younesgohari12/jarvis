from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis.memory.store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_persistence_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-memory-") as temporary:
            path = Path(temporary) / "memory" / "jarvis.db"
            first = MemoryStore(path)
            session_id, resumed = first.start_session(24)
            self.assertFalse(resumed)
            first.add_message(session_id, "user", "اسم من یونس است")
            first.add_message(session_id, "assistant", "ثبت شد")
            first.set_fact("user_name", "یونس")
            first.set_setting("personality", "Kind")
            first.close()

            second = MemoryStore(path)
            resumed_id, resumed = second.start_session(24)
            self.assertTrue(resumed)
            self.assertEqual(resumed_id, session_id)
            self.assertEqual(second.get_fact("user_name"), "یونس")
            self.assertEqual(second.get_setting("personality"), "Kind")
            self.assertEqual([item.role for item in second.recent_messages(session_id)], ["user", "assistant"])
            self.assertTrue(second.integrity_check())
            second.close()

    def test_session_context_is_structured_and_isolated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-memory-") as temporary:
            store = MemoryStore(Path(temporary) / "jarvis.db")
            first, _ = store.start_session()
            store.set_context(first, "project", {"pending": "project_kind", "technology": "python"})
            second = store.create_new_session()
            self.assertEqual(store.get_context(first)["technology"], "python")
            self.assertEqual(store.get_context(first)["topic"], "project")
            self.assertEqual(store.get_context(second), {"topic": ""})
            store.close()

    def test_attachment_metadata_is_remembered(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-memory-") as temporary:
            store = MemoryStore(Path(temporary) / "jarvis.db")
            session, _ = store.start_session()
            store.remember_attachment(session, "sample.zip", "C:/tmp/sample.zip", "zip", 123)
            attachment = store.last_attachment(session)
            self.assertIsNotNone(attachment)
            assert attachment is not None
            self.assertEqual(attachment.name, "sample.zip")
            self.assertEqual(store.get_fact("last_file"), "sample.zip")
            store.close()

    def test_boolean_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-memory-") as temporary:
            store = MemoryStore(Path(temporary) / "jarvis.db")
            store.set_setting("internet_enabled", True)
            store.set_setting("animation", False)
            self.assertTrue(store.get_bool_setting("internet_enabled"))
            self.assertFalse(store.get_bool_setting("animation", True))
            store.close()

    def test_cleanup_enforces_message_and_session_bounds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-memory-") as temporary:
            store = MemoryStore(
                Path(temporary) / "jarvis.db",
                max_history_rows=100,
                max_sessions=5,
                retention_days=90,
            )
            session, _ = store.start_session()
            for _ in range(8):
                session = store.create_new_session()
            for index in range(150):
                store.add_message(session, "user", f"message {index}")
            result = store.cleanup()
            self.assertLessEqual(result["sessions_remaining"], 5)
            self.assertLessEqual(len(store.recent_messages(session, 1000)), 100)
            store.close()

    def test_clear_long_term_memory_preserves_settings(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-memory-") as temporary:
            store = MemoryStore(Path(temporary) / "jarvis.db")
            session, _ = store.start_session()
            store.set_fact("user_name", "Younes")
            store.set_context(session, "project", {"technology": "python"})
            store.set_setting("theme", "CYBER")
            store.clear_long_term_memory()
            self.assertIsNone(store.get_fact("user_name"))
            self.assertEqual(store.get_context(session), {"topic": ""})
            self.assertEqual(store.get_setting("theme"), "CYBER")
            store.close()

    def test_new_session_is_distinct(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-memory-") as temporary:
            store = MemoryStore(Path(temporary) / "jarvis.db")
            first, _ = store.start_session(24)
            second = store.create_new_session()
            self.assertNotEqual(first, second)
            store.close()


if __name__ == "__main__":
    unittest.main()
