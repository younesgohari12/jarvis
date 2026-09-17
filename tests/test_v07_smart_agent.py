from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from jarvis.agent.permissions import PermissionLayer
from jarvis.agent.task_context import TaskContext
from jarvis.knowledge.rag import RAGStore
from jarvis.lab.catalog import DatasetCatalog, ModelCatalog
from jarvis.memory.store import MemoryStore
from jarvis.neural.tokenizer import JarvisTokenizer, normalize_language_text
from jarvis.neural.transformer import JarvisTransformer, NeuralModelError
from jarvis.tools.files import FileManager
from tests.helpers import TemporaryRuntime


ROOT = Path(__file__).resolve().parents[1]


class SmartAgentV07Tests(unittest.TestCase):
    def test_release_assets_are_v2_quantized_and_integrity_checked(self) -> None:
        model = JarvisTransformer.load(ROOT / "models" / "jarvis_nano_v07.npz")
        self.assertEqual(model.parameter_count, 23_077_376)
        self.assertEqual(model.config.max_seq_len, 2048)
        self.assertEqual(model.metadata["model_format"], "jarvis-numpy-int8-v2")
        self.assertEqual(model.metadata["quantization"], "symmetric_per_tensor_int8")
        self.assertTrue(model.quantized_runtime)
        self.assertLess(model.memory_bytes, 24 * 1024 * 1024)
        self.assertTrue(any(value.dtype == np.int8 for value in model.parameters.values()))
        self.assertIsNone(model.metadata["pretrained_source"])
        self.assertTrue(model.metadata["integrity"]["weights_sha256"])

    def test_v2_model_rejects_tampered_parameter(self) -> None:
        source = ROOT / "models" / "jarvis_nano_v07.npz"
        with tempfile.TemporaryDirectory(prefix="jarvis-v7-integrity-") as temporary:
            target = Path(temporary) / "tampered.npz"
            with np.load(source, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            parameter = next(name for name in arrays if not name.startswith("__") and arrays[name].dtype == np.int8)
            arrays[parameter].flat[0] = np.int8(int(arrays[parameter].flat[0]) ^ 1)
            np.savez_compressed(target, **arrays)
            with self.assertRaises(NeuralModelError):
                JarvisTransformer.load(target)

    def test_smart_brain_is_lazy_at_startup(self) -> None:
        with TemporaryRuntime() as runtime:
            self.assertIsNotNone(runtime.neural)
            self.assertFalse(runtime.neural.loaded)
            self.assertEqual(runtime.neural.parameter_count, 23_077_376)
            runtime.agent.respond("hello")
            self.assertFalse(runtime.neural.loaded)

    def test_tokenizer_round_trips_persian_paths_urls_json_and_code(self) -> None:
        tokenizer = JarvisTokenizer.load(ROOT / "models" / "jarvis_tokenizer_v003.json")
        samples = (
            "داخل درایو D یک فایل به نام تست بساز",
            r"D:\Projects\Jarvis\config\app.json",
            "https://example.com/api?q=سلام&limit=20",
            '{"tool":"create_file","arguments":{"path":"D:\\test.txt"}}',
            "def hello(name: str) -> str:\n    return f'سلام {name}'",
        )
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), normalize_language_text(text))

    def test_exact_drive_file_request_defaults_to_txt(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("داخل درایو d یک فایل به نام تست بساز")
            self.assertEqual(route.intent, "create_file")
            self.assertEqual(route.arguments, {"path": "D:\\تست.txt", "content": ""})
            plan = runtime.agent.planner.plan("request", route)
            self.assertIsNotNone(plan)
            self.assertTrue(plan.needs_confirmation)  # type: ignore[union-attr]

    def test_exact_drive_folder_request_has_no_fake_extension(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("داخل drive E فولدری موسوم به پروژه ایجاد کن")
            self.assertEqual(route.intent, "create_folder")
            self.assertEqual(route.arguments["path"], "E:\\پروژه")

    def test_contextual_same_folder_reference_is_resolved(self) -> None:
        with TemporaryRuntime() as runtime:
            context = TaskContext.update({}, "open_folder", {"path": "D:\\Work"})
            context["_planning_chain"] = True
            route = runtime.agent.router.route("همونجا فایل notes بساز", context)
            self.assertEqual(route.intent, "create_file")
            self.assertEqual(route.arguments["path"], "D:\\Work\\notes.txt")
            self.assertTrue(route.referenced)

    def test_dynamic_pdf_workflow_composes_search_select_move(self) -> None:
        with TemporaryRuntime() as runtime:
            prompt = "فایل‌های PDF امروزی Downloads رو پیدا کن؛ حجیم‌ترینش رو ببر Desktop"
            route = runtime.agent.router.route(prompt)
            plan = runtime.agent.planner.plan(prompt, route)
            self.assertEqual(route.intent, "file_selection_workflow")
            self.assertIsNotNone(plan)
            calls = [step.tool_call for step in plan.steps if step.tool_call]  # type: ignore[union-attr]
            self.assertEqual([call.tool for call in calls], ["search_files", "move_file"])
            self.assertEqual(calls[1].arguments["source"], {"$ref": "step1.items.0.path"})
            self.assertTrue(calls[1].requires_confirmation)

    def test_educational_deletion_question_never_becomes_a_tool_plan(self) -> None:
        with TemporaryRuntime() as runtime:
            prompt = "پاک کردن فایل در پایتون چطور انجام میشه؟"
            route = runtime.agent.router.route(prompt)
            self.assertEqual(route.intent, "knowledge_question")
            self.assertIsNone(runtime.agent.planner.plan(prompt, route))
            reply = runtime.agent.respond(prompt)
            self.assertNotIn(reply.intent, {"delete_file", "tool_confirm_dangerous"})

    def test_drive_delete_request_is_protected(self) -> None:
        with TemporaryRuntime() as runtime:
            route = runtime.agent.router.route("درایو C رو پاک کن")
            self.assertEqual(route.intent, "dangerous_request")
            self.assertTrue(route.requires_confirmation)

    def test_dry_run_never_mutates_disk(self) -> None:
        with TemporaryRuntime() as runtime, tempfile.TemporaryDirectory(prefix="jarvis-v7-dry-") as temporary:
            target = Path(temporary) / "dry.txt"
            reply = runtime.agent.respond(f"dry run: create file {target}")
            self.assertEqual(reply.intent, "tool_dry_run")
            self.assertFalse(target.exists())
            self.assertFalse(reply.data["executed"])

    def test_confirmed_create_can_be_undone(self) -> None:
        with TemporaryRuntime() as runtime, tempfile.TemporaryDirectory(prefix="jarvis-v7-undo-") as temporary:
            target = Path(temporary) / "undo.txt"
            self.assertEqual(runtime.agent.respond(f"create file {target}").intent, "tool_confirm_dangerous")
            self.assertEqual(runtime.agent.respond("yes").intent, "tool_file_operation_result")
            self.assertTrue(target.exists())
            self.assertEqual(runtime.agent.respond("undo").intent, "tool_confirm_dangerous")
            self.assertEqual(runtime.agent.respond("yes").intent, "tool_undo_result")
            self.assertFalse(target.exists())

    def test_permission_levels_cover_l0_through_l4(self) -> None:
        expected = {
            "information": "L0", "external_app": "L1", "write_file": "L2",
            "delete": "L3", "power_control": "L4",
        }
        for category, code in expected.items():
            with self.subTest(category=category):
                self.assertEqual(PermissionLayer.level_for(category).code, code)  # type: ignore[union-attr]

    def test_typo_tolerance_and_persian_number_understanding(self) -> None:
        with TemporaryRuntime() as runtime:
            open_route = runtime.agent.router.route("کروم رو با زکن")
            move_route = runtime.agent.router.route("فایلو انتغال بده")
            volume_route = runtime.agent.router.route("صدا رو بکن پنجاه درصد")
            self.assertEqual((open_route.intent, open_route.arguments.get("app")), ("open_app", "chrome"))
            self.assertEqual(move_route.intent, "move_file")
            self.assertEqual(volume_route.arguments, {"percent": 50})

    def test_diagnostics_are_read_only_and_report_real_assets(self) -> None:
        with TemporaryRuntime() as runtime:
            result = runtime.tools.invoke("diagnose_system", {"scope": "jarvis"})
            self.assertEqual(result["status"], "healthy")
            checks = {row["name"]: row for row in result["checks"]}
            self.assertTrue(checks["memory_database"]["healthy"])
            self.assertTrue(checks["model_assets"]["healthy"])

    def test_skill_discovery_validates_required_tools(self) -> None:
        with TemporaryRuntime() as runtime:
            report = runtime.skills.report()
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["discovered"], 3)
            self.assertEqual(report["available"], 3)
            self.assertIn("list_skills", runtime.tools.names())
            plugins = runtime.plugins.report()
            self.assertEqual(plugins["discovered"], 1)
            self.assertEqual(plugins["available"], 1)
            self.assertEqual(runtime.agent.router.route("list plugins").intent, "list_plugins")

    def test_three_layer_memory_conflict_consolidation_and_forgetting(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-v7-memory-") as temporary:
            store = MemoryStore(Path(temporary) / "memory.db")
            session, _ = store.start_session()
            store.add_message(session, "user", "hello")
            store.set_fact("user_name", "یونس")
            store.set_fact("user_name", "Younes")
            store.record_episode(session, "ساخت فایل پروژه", 0.8)
            report = store.consolidate_memory(20)
            self.assertEqual(store.semantic_memories()["user_name"], "Younes")
            self.assertGreaterEqual(report["semantic_conflicts"], 1)
            self.assertEqual(store.episodes()[0].summary, "ساخت فایل پروژه")
            self.assertGreaterEqual(store.forget("semantic", "user_name"), 1)
            self.assertIsNone(store.get_fact("user_name"))
            self.assertTrue(store.integrity_check())
            store.close()

    def test_hybrid_rag_reports_scoring_boundaries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-v7-rag-") as temporary:
            store = RAGStore(Path(temporary) / "rag.db")
            store.index_text(
                Path(temporary) / "network.txt",
                "TCP اتصال‌گرا و قابل اعتماد است. UDP بدون اتصال و کم‌تاخیر است.",
            )
            hits = store.search("تفاوت TCP و UDP", 3)
            self.assertTrue(hits)
            hit = hits[0]
            self.assertGreater(hit.bm25_score, 0.0)
            self.assertGreater(hit.vector_score, 0.0)
            self.assertGreater(hit.rerank_score, 0.0)
            self.assertEqual(hit.source, "local_document")
            store.close()

    def test_txt_and_zip_attachment_paths_are_bounded_and_readable(self) -> None:
        with TemporaryRuntime() as runtime, tempfile.TemporaryDirectory(prefix="jarvis-v7-files-") as temporary:
            manager = FileManager(runtime.config.files)
            root = Path(temporary)
            text_path = root / "سلام.txt"
            text_path.write_text("سلام JARVIS", encoding="utf-8")
            self.assertIn("سلام JARVIS", manager.inspect(text_path).preview)
            zip_path = root / "bundle.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("docs/readme.md", "# JARVIS\nمتن فارسی")
                archive.writestr("src/main.py", "print('ok')")
            inspection = manager.inspect(zip_path)
            self.assertEqual([entry.name for entry in inspection.zip_entries], ["docs/readme.md", "src/main.py"])
            self.assertIn("متن فارسی", manager.read_zip_text_member(zip_path, "docs/readme.md"))

    def test_unseen_benchmark_is_separate_and_has_one_thousand_cases(self) -> None:
        rows = [
            json.loads(line)
            for line in (ROOT / "datasets" / "benchmarks" / "unseen_v003_1000.jsonl")
            .read_text(encoding="utf-8").splitlines()
        ]
        manifest = json.loads((ROOT / "datasets" / "manifest_v003.json").read_text(encoding="utf-8"))
        evaluation = json.loads((ROOT / "models" / "agent_evaluation_v7.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 1000)
        self.assertTrue(all(row["seen_in_training"] is False for row in rows))
        self.assertEqual(manifest["unseen_benchmark_training_overlap"], 0)
        self.assertEqual(evaluation["dataset_cases"], 1000)
        self.assertEqual(evaluation["error_count"], 0)

    def test_lab_dataset_catalog_finds_duplicates_and_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-v7-catalog-") as temporary:
            root = Path(temporary)
            path = root / "datasets" / "raw" / "sample.jsonl"
            path.parent.mkdir(parents=True)
            valid = {"input": "سلام", "output": "درود", "category": "conversation", "language": "fa"}
            path.write_text(
                json.dumps(valid, ensure_ascii=False) + "\n"
                + json.dumps(valid, ensure_ascii=False) + "\n{broken\n",
                encoding="utf-8",
            )
            record = DatasetCatalog(root).validate(path)
            self.assertEqual((record.samples, record.duplicates, record.malformed), (2, 1, 1))
            self.assertEqual(record.categories, ("conversation",))

    def test_lab_dataset_import_toggle_and_bounded_delete(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-v7-import-") as temporary:
            base = Path(temporary)
            root = base / "project"
            (root / "datasets" / "raw").mkdir(parents=True)
            protected = root / "datasets" / "raw" / "protected.jsonl"
            protected.write_text('{"input":"a","output":"b"}\n', encoding="utf-8")
            source = base / "source.jsonl"
            source.write_text('{"input":"new","output":"target","language":"en"}\n', encoding="utf-8")
            catalog = DatasetCatalog(root)
            imported = catalog.import_jsonl(source)
            catalog.set_enabled(imported.path, False)
            self.assertFalse(catalog.validate(root / imported.path).enabled)
            with self.assertRaises(PermissionError):
                catalog.delete_imported("datasets/raw/protected.jsonl")
            catalog.delete_imported(imported.path)
            self.assertFalse((root / imported.path).exists())
            self.assertTrue(protected.exists())

    def test_lab_model_catalog_protects_bundled_models(self) -> None:
        catalog = ModelCatalog(ROOT)
        records = {record.key: record for record in catalog.scan()}
        self.assertTrue(records["jarvis_nano_v08"].activatable)
        self.assertFalse(records["jarvis_core_v07"].activatable)
        with self.assertRaises(PermissionError):
            catalog.delete_lab_model("models/jarvis_nano_v08.npz")


if __name__ == "__main__":
    unittest.main()
