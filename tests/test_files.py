from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from jarvis.config import FileLimits
from jarvis.tools.files import FileManager, FileToolError, SUPPORTED_EXTENSIONS


class FileManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = FileLimits(
            max_file_bytes=50_000,
            max_text_preview_bytes=64,
            max_zip_entries=20,
            max_zip_member_bytes=128,
            max_zip_total_preview_bytes=512,
        )
        self.manager = FileManager(self.limits)

    def test_required_extensions_are_supported(self) -> None:
        expected = {
            ".txt", ".py", ".js", ".ts", ".json", ".md", ".csv", ".html",
            ".css", ".xml", ".log", ".pdf", ".docx", ".xlsx", ".zip",
        }
        self.assertEqual(SUPPORTED_EXTENSIONS, expected)

    def test_text_attach_is_bounded_and_unicode_safe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-files-") as temporary:
            path = Path(temporary) / "sample.txt"
            path.write_text("سلام جارویس\n" + "x" * 100, encoding="utf-8")
            result = self.manager.inspect(path)
            self.assertEqual(result.kind, "text")
            self.assertIn("سلام جارویس", result.preview)
            self.assertTrue(result.preview_truncated)

    def test_zip_lists_and_reads_text_without_extracting(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-files-") as temporary:
            root = Path(temporary)
            path = root / "sample.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("docs/readme.md", "# Hello\nسلام")
                archive.writestr("../unsafe.txt", "not extracted")
                archive.writestr("assets/image.bin", b"\x00\x01")
            result = self.manager.inspect(path)
            self.assertEqual(result.kind, "zip")
            self.assertEqual(len(result.zip_entries), 3)
            self.assertIn("# Hello", self.manager.read_zip_text_member(path, "docs/readme.md"))
            self.assertFalse((root.parent / "unsafe.txt").exists())
            self.assertIn("unsafe.txt", [entry.name for entry in result.zip_entries])

    def test_zip_entry_count_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-files-") as temporary:
            path = Path(temporary) / "many.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for index in range(30):
                    archive.writestr(f"item-{index}.txt", str(index))
            result = self.manager.inspect(path)
            self.assertEqual(len(result.zip_entries), 20)
            self.assertTrue(result.entries_truncated)

    def test_large_zip_member_cannot_be_read_into_ram(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-files-") as temporary:
            path = Path(temporary) / "large-member.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("large.txt", "x" * 500)
            with self.assertRaises(FileToolError):
                self.manager.read_zip_text_member(path, "large.txt")

    def test_oversized_file_is_rejected_before_reading(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-files-") as temporary:
            path = Path(temporary) / "huge.log"
            path.write_bytes(b"x" * 50_001)
            with self.assertRaises(FileToolError):
                self.manager.inspect(path)

    def test_invalid_zip_and_unsupported_extension_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-files-") as temporary:
            root = Path(temporary)
            broken = root / "broken.zip"
            broken.write_bytes(b"not-a-zip")
            executable = root / "program.exe"
            executable.write_bytes(b"MZ")
            with self.assertRaises(FileToolError):
                self.manager.inspect(broken)
            with self.assertRaises(FileToolError):
                self.manager.inspect(executable)

    def test_binary_zip_member_is_not_read_as_text(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-files-") as temporary:
            path = Path(temporary) / "binary.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("image.bin", b"\x00\x01")
            with self.assertRaises(FileToolError):
                self.manager.read_zip_text_member(path, "image.bin")


if __name__ == "__main__":
    unittest.main()
