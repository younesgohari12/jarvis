from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from jarvis.knowledge.rag import RAGStore
from jarvis.learning.queue import LearningQueue
from jarvis.monitoring.live import LiveMonitor
from jarvis.neural.transformer import JarvisTransformer
from jarvis.research.engine import ResearchEngine
from jarvis.runtime.hardware import HardwareManager
from jarvis.search.engine import SearchEvidence, SearchReport
from tests.helpers import TemporaryRuntime


class _SearchStub:
    def search(self, query: str) -> SearchReport:
        evidence = SearchEvidence(
            "Official Python documentation", "https://docs.python.org/3/",
            "Python documentation describes the language and standard library in detail.", 0.8,
        )
        return SearchReport(query, "Python is a programming language with a broad standard library.", (evidence,), 0)


class V06SystemsTests(unittest.TestCase):
    def test_rag_indexes_and_retrieves_local_grounding(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-rag-") as temporary:
            store = RAGStore(Path(temporary) / "rag.db", 20)
            try:
                store.index_text(Path(temporary) / "notes.txt", "پروژه آلفا از SQLite برای حافظه محلی استفاده می‌کند.")
                hits = store.search("حافظه پروژه آلفا")
                self.assertTrue(hits)
                self.assertIn("SQLite", hits[0].chunk)
                self.assertEqual(store.stats(), {"documents": 1, "chunks": 1})
            finally:
                store.close()

    def test_learning_queue_requires_review_and_never_trains(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-learning-") as temporary:
            queue = LearningQueue(Path(temporary) / "queue.jsonl")
            row = queue.add(
                original_input="سلام", jarvis_response="درود",
                correct_response="سلام یونس", notes="preferred greeting",
            )
            self.assertEqual(queue.rows("pending")[0]["id"], row["id"])
            self.assertTrue(queue.set_status(row["id"], "approved"))
            self.assertEqual(queue.rows("approved")[0]["correct_response"], "سلام یونس")

    def test_research_has_real_sources_and_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-research-") as temporary:
            log = Path(temporary) / "search.jsonl"
            report = ResearchEngine(_SearchStub(), log).research("Python")  # type: ignore[arg-type]
            self.assertTrue(report.summary)
            self.assertEqual(report.sources[0].url, "https://docs.python.org/3/")
            event = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["selected_sources"], ["https://docs.python.org/3/"])

    def test_live_monitor_reports_cpu_ram_disk_without_gpu_fabrication(self) -> None:
        monitor = LiveMonitor(HardwareManager())
        snapshot = monitor.sample()
        self.assertGreaterEqual(snapshot.cpu_percent, 0)
        self.assertGreater(snapshot.ram_total_mb, 0)
        self.assertGreaterEqual(snapshot.disk_free_gb, 0)
        if not snapshot.gpu_available:
            self.assertIsNone(snapshot.gpu_percent)
            self.assertIsNone(snapshot.vram_used_mb)

    def test_v06_routes_and_personality_switching(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("مود رو عصبانی کن")
            self.assertEqual(route.intent, "set_personality")
            reply = runtime.agent.respond("مود رو عصبانی کن")
            self.assertEqual(runtime.agent.current_personality, "Angry")
            self.assertEqual(reply.intent, "personality_changed")
            research = runtime.agent.router.route("درباره پایتون تحقیق کن")
            self.assertEqual(research.intent, "web_research")
            folder = runtime.agent.router.route("فولدر Younes Gohari رو داخل D پیدا کن")
            self.assertIn(folder.intent, {"find_folder", "find_and_open_folder"})
            self.assertEqual(folder.arguments.get("root"), "D:\\")

    def test_release_has_real_stage_checkpoints(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = (
            "stage1_language_foundations", "stage2_conversation",
            "stage3_general_knowledge", "stage4_instruction_following",
            "stage5_action_recognition", "stage6_entity_extraction",
            "stage7_tool_calling", "stage8_argument_extraction",
            "stage9_context_memory", "stage10_planning", "stage11_reasoning",
            "stage12_recovery_verification", "stage13_search_decision",
            "stage14_jarvis_personality",
        )
        for stage in expected:
            directory = root / "models" / "checkpoints_v7" / stage
            manifest = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertIsNone(manifest.get("pretrained_source"))
            self.assertGreaterEqual(manifest["size_bytes"], 0)
        # The release retention policy keeps every stage manifest but only the
        # final trained weight, avoiding 14 nearly identical 15 MB archives.
        final_manifest_path = root / "models" / "checkpoints_v7" / "stage14_jarvis_personality" / "checkpoint.json"
        final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        final = (final_manifest_path.parent / final_manifest["model_file"]).resolve()
        self.assertTrue(final.is_file())
        self.assertEqual(final, (root / "models" / "jarvis_nano_v07.npz").resolve())
        self.assertEqual(JarvisTransformer.file_sha256(final), final_manifest["sha256"])


if __name__ == "__main__":
    unittest.main()
