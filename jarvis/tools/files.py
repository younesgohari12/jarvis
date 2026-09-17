from __future__ import annotations

import mimetypes
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from jarvis.config import FileLimits


SUPPORTED_TEXT_EXTENSIONS = frozenset(
    {".txt", ".py", ".js", ".ts", ".json", ".md", ".csv", ".html", ".css", ".xml", ".log"}
)
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx"})
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS | {".zip"}


class FileToolError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ZipEntry:
    name: str
    size: int
    compressed_size: int
    is_directory: bool
    text_preview: str | None = None
    preview_truncated: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class FileInspection:
    path: Path
    name: str
    extension: str
    mime_type: str
    size: int
    kind: str
    preview: str = ""
    preview_truncated: bool = False
    zip_entries: tuple[ZipEntry, ...] = ()
    entries_truncated: bool = False

    @staticmethod
    def _size_label(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.2f} MB"

    def display(self, language: str = "fa") -> str:
        if language == "en":
            lines = [
                f"Attached: {self.name}",
                f"Size: {self._size_label(self.size)} | Type: {self.extension or self.mime_type}",
            ]
            if self.kind == "text":
                lines.extend(["", "Text preview:", self.preview or "(empty file)"])
                if self.preview_truncated:
                    lines.append("[Preview truncated to protect memory]")
            else:
                lines.extend(["", f"ZIP structure ({len(self.zip_entries)} shown):"])
                lines.extend(self._zip_lines())
                if self.entries_truncated:
                    lines.append("[More entries were omitted]")
            return "\n".join(lines)

        lines = [
            f"فایل پیوست شد: {self.name}",
            f"حجم: {self._size_label(self.size)} | نوع: {self.extension or self.mime_type}",
        ]
        if self.kind == "text":
            lines.extend(["", "پیش‌نمایش متن:", self.preview or "(فایل خالی است)"])
            if self.preview_truncated:
                lines.append("[برای محافظت از RAM، پیش‌نمایش کوتاه شده است]")
        else:
            lines.extend(["", f"ساختار ZIP ({len(self.zip_entries)} مورد نمایش داده شده):"])
            lines.extend(self._zip_lines())
            if self.entries_truncated:
                lines.append("[موارد بیشتری وجود داشت و نمایش داده نشد]")
        return "\n".join(lines)

    def _zip_lines(self) -> list[str]:
        lines: list[str] = []
        for entry in self.zip_entries:
            icon = "📁" if entry.is_directory else "📄"
            size = "" if entry.is_directory else f" ({self._size_label(entry.size)})"
            lines.append(f"{icon} {entry.name}{size}")
            if entry.text_preview:
                preview = entry.text_preview
                if len(preview) > 1200:
                    preview = preview[:1199].rstrip() + "…"
                lines.append(f"    └─ {preview.replace(chr(10), chr(10) + '       ')}")
            if entry.note:
                lines.append(f"    [{entry.note}]")
        return lines


@dataclass(frozen=True, slots=True)
class FolderEntry:
    name: str
    path: Path
    is_directory: bool
    size: int


@dataclass(frozen=True, slots=True)
class FolderListing:
    path: Path
    entries: tuple[FolderEntry, ...]
    truncated: bool = False

    def display(self, language: str = "fa") -> str:
        title = f"Folder: {self.path}" if language == "en" else f"پوشه: {self.path}"
        lines = [title]
        for entry in self.entries:
            marker = "📁" if entry.is_directory else "📄"
            lines.append(f"{marker} {entry.name}")
        if self.truncated:
            lines.append("[More items omitted]" if language == "en" else "[موارد بیشتری نمایش داده نشد]")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FileOperationResult:
    success: bool
    action: str
    source: Path | None
    destination: Path
    bytes_written: int = 0
    code: str = ""
    verified: bool = True
    previous_size: int | None = None


class FileManager:
    def __init__(self, limits: FileLimits) -> None:
        self.limits = limits

    def inspect(self, raw_path: str | Path) -> FileInspection:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileToolError("File does not exist")
        if not path.is_file():
            raise FileToolError("Selected path is not a file")
        extension = path.suffix.casefold()
        if extension not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise FileToolError(f"Unsupported file type. Supported: {supported}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise FileToolError(f"Cannot read file information: {exc}") from exc
        if size > self.limits.max_file_bytes:
            maximum = self.limits.max_file_bytes / (1024 * 1024)
            raise FileToolError(f"File is larger than the {maximum:.0f} MB safety limit")

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if extension == ".zip":
            return self._inspect_zip(path, size, mime_type)
        if extension == ".pdf":
            return self._inspect_pdf(path, size, mime_type)
        if extension == ".docx":
            return self._inspect_docx(path, size, mime_type)
        if extension == ".xlsx":
            return self._inspect_xlsx(path, size, mime_type)
        return self._inspect_text(path, size, extension, mime_type)

    def list_folder(self, raw_path: str | Path, limit: int = 250) -> FolderListing:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileToolError("Folder does not exist")
        maximum = max(1, min(500, int(limit)))
        try:
            children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError as exc:
            raise FileToolError(f"Cannot list folder: {exc}") from exc
        entries: list[FolderEntry] = []
        for child in children[:maximum]:
            try:
                entries.append(
                    FolderEntry(child.name, child, child.is_dir(), 0 if child.is_dir() else child.stat().st_size)
                )
            except OSError:
                entries.append(FolderEntry(child.name, child, child.is_dir(), 0))
        return FolderListing(path, tuple(entries), len(children) > maximum)

    def find_file(
        self, raw_folder: str | Path, name: str, maximum_results: int = 20
    ) -> tuple[Path, ...]:
        root = Path(raw_folder).expanduser().resolve()
        if not root.is_dir():
            raise FileToolError("Search folder does not exist")
        needle = str(name).strip().casefold()
        if not needle or len(needle) > 180:
            raise FileToolError("A valid filename is required")
        results: list[Path] = []
        visited = 0
        try:
            for folder, directories, files in os.walk(root):
                visited += 1
                if visited > 800 or len(results) >= max(1, min(50, maximum_results)):
                    break
                directories[:] = [value for value in directories if not value.startswith(".")][:80]
                for filename in files:
                    if needle in filename.casefold():
                        results.append(Path(folder) / filename)
                        if len(results) >= maximum_results:
                            break
        except OSError as exc:
            raise FileToolError(f"Cannot search folder: {exc}") from exc
        return tuple(results)

    def search_files(
        self,
        raw_root: str | Path,
        *,
        extension: str = "",
        modified: str = "any",
        order_by: str = "modified",
        descending: bool = True,
        limit: int = 20,
    ) -> dict[str, object]:
        """Bounded metadata search used by dynamic desktop plans."""
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileToolError("Search folder does not exist")
        suffix = str(extension).strip().casefold()
        if suffix and not suffix.startswith("."):
            suffix = "." + suffix
        if len(suffix) > 16 or any(value in suffix for value in ("/", "\\", "*", "?")):
            raise FileToolError("Invalid extension filter")
        period = str(modified).casefold().strip()
        if period not in {"any", "today", "yesterday"}:
            raise FileToolError("Unsupported modified-date filter")
        sort_key = str(order_by).casefold().strip()
        if sort_key not in {"name", "size", "modified"}:
            raise FileToolError("Unsupported file sort key")
        maximum = max(1, min(100, int(limit)))
        target_date: date | None = None
        if period == "today":
            target_date = date.today()
        elif period == "yesterday":
            target_date = date.today() - timedelta(days=1)

        items: list[dict[str, object]] = []
        visited = scanned = 0
        truncated = False
        try:
            for folder, directories, filenames in os.walk(root):
                visited += 1
                if visited > 1_200 or scanned > 25_000:
                    truncated = True
                    break
                directories[:] = [name for name in directories if not name.startswith(".")][:100]
                for filename in filenames:
                    scanned += 1
                    candidate = Path(folder) / filename
                    if suffix and candidate.suffix.casefold() != suffix:
                        continue
                    try:
                        stat = candidate.stat()
                    except OSError:
                        continue
                    modified_at = datetime.fromtimestamp(stat.st_mtime)
                    if target_date is not None and modified_at.date() != target_date:
                        continue
                    items.append(
                        {
                            "path": str(candidate),
                            "name": candidate.name,
                            "extension": candidate.suffix.casefold(),
                            "size": int(stat.st_size),
                            "modified": modified_at.isoformat(timespec="seconds"),
                            "modified_timestamp": float(stat.st_mtime),
                        }
                    )
        except OSError as exc:
            raise FileToolError(f"Cannot search folder: {exc}") from exc

        key = {
            "name": lambda item: str(item["name"]).casefold(),
            "size": lambda item: int(item["size"]),
            "modified": lambda item: float(item["modified_timestamp"]),
        }[sort_key]
        items.sort(key=key, reverse=bool(descending))
        selected = items[:maximum]
        return {
            "status": "ok" if selected else "not_found",
            "root": str(root),
            "filters": {"extension": suffix, "modified": period},
            "order_by": sort_key,
            "descending": bool(descending),
            "items": selected,
            "matched": len(items),
            "scanned": scanned,
            "truncated": truncated or len(items) > maximum,
        }

    def find_folder(
        self, raw_root: str | Path, name: str, maximum_results: int = 20
    ) -> tuple[Path, ...]:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileToolError("Search root does not exist")
        needle = str(name).strip().casefold()
        if not needle or len(needle) > 180:
            raise FileToolError("A valid folder name is required")
        scored: list[tuple[float, Path]] = []
        visited = 0
        try:
            for folder, directories, _files in os.walk(root):
                visited += 1
                if visited > 1200:
                    break
                visible = [value for value in directories if not value.startswith(".")][:100]
                directories[:] = visible
                for directory in visible:
                    candidate = directory.casefold()
                    score = 1.0 if candidate == needle else 0.92 if needle in candidate else SequenceMatcher(None, needle, candidate).ratio()
                    if score >= 0.62:
                        scored.append((score, Path(folder) / directory))
        except OSError as exc:
            raise FileToolError(f"Cannot search folders: {exc}") from exc
        scored.sort(key=lambda item: (-item[0], len(item[1].parts), item[1].name.casefold()))
        return tuple(path for _score, path in scored[: max(1, min(50, maximum_results))])

    def write_text(self, raw_path: str | Path, content: str) -> FileOperationResult:
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.casefold() not in SUPPORTED_TEXT_EXTENSIONS:
            raise FileToolError("Writing is limited to supported text files")
        data = str(content).encode("utf-8")
        if len(data) > self.limits.max_text_preview_bytes * 8:
            raise FileToolError("Text exceeds the safe write limit")
        if path.exists():
            raise FileToolError("Destination already exists; use append or choose a new file")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(data)
        except OSError as exc:
            raise FileToolError(f"Cannot write file: {exc}") from exc
        return FileOperationResult(True, "write", None, path, len(data), "written")

    def create_file(self, raw_path: str | Path, content: str = "") -> FileOperationResult:
        path = Path(raw_path).expanduser().resolve()
        if path.exists():
            raise FileToolError("File already exists")
        result = self.write_text(path, content)
        return FileOperationResult(
            result.success, "create_file", None, result.destination,
            result.bytes_written, "created", result.destination.is_file(),
        )

    def append_text(self, raw_path: str | Path, content: str) -> FileOperationResult:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_TEXT_EXTENSIONS:
            raise FileToolError("Appending is limited to an existing supported text file")
        data = str(content).encode("utf-8")
        if not data or len(data) > self.limits.max_text_preview_bytes * 4:
            raise FileToolError("Text is empty or exceeds the safe append limit")
        previous_size = path.stat().st_size
        if previous_size + len(data) > self.limits.max_file_bytes:
            raise FileToolError("Append would exceed the file safety limit")
        try:
            with path.open("ab") as handle:
                handle.write(data)
        except OSError as exc:
            raise FileToolError(f"Cannot append file: {exc}") from exc
        return FileOperationResult(
            True, "append", path, path, len(data), "appended", True, previous_size
        )

    def truncate_file(self, raw_path: str | Path, size: int) -> FileOperationResult:
        """Internal reversible-operation helper; invocation still uses central permissions."""
        path = Path(raw_path).expanduser().resolve()
        target_size = int(size)
        if not path.is_file() or target_size < 0 or target_size > path.stat().st_size:
            raise FileToolError("A valid file and smaller target size are required")
        try:
            with path.open("r+b") as handle:
                handle.truncate(target_size)
        except OSError as exc:
            raise FileToolError(f"Cannot restore file size: {exc}") from exc
        return FileOperationResult(
            True, "truncate", path, path, 0, "restored", path.stat().st_size == target_size
        )

    def create_folder(self, raw_path: str | Path) -> FileOperationResult:
        path = Path(raw_path).expanduser().resolve()
        if path.exists():
            raise FileToolError("Folder or file already exists")
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise FileToolError(f"Cannot create folder: {exc}") from exc
        return FileOperationResult(True, "create_folder", None, path, 0, "created", path.is_dir())

    @staticmethod
    def _deletion_target(raw_path: str | Path, *, folder: bool) -> Path:
        raw = Path(raw_path).expanduser()
        path = raw.absolute() if raw.is_symlink() else raw.resolve()
        anchors = {Path(path.anchor)} if path.anchor else set()
        protected = {Path.home().resolve()}
        if os.name == "nt":
            for variable in ("WINDIR", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
                value = os.environ.get(variable)
                if value:
                    protected.add(Path(value).resolve())
        if path in anchors or path in protected:
            raise FileToolError("Protected root/system path cannot be deleted")
        if folder and not path.is_dir():
            raise FileToolError("Folder does not exist")
        if not folder and not (path.is_file() or path.is_symlink()):
            raise FileToolError("File does not exist")
        return path

    def delete_file(self, raw_path: str | Path) -> FileOperationResult:
        path = self._deletion_target(raw_path, folder=False)
        try:
            path.unlink()
        except OSError as exc:
            raise FileToolError(f"Cannot delete file: {exc}") from exc
        return FileOperationResult(True, "delete_file", path, path, 0, "deleted", not path.exists())

    def delete_folder(self, raw_path: str | Path) -> FileOperationResult:
        path = self._deletion_target(raw_path, folder=True)
        try:
            path.rmdir()
        except OSError as exc:
            raise FileToolError(
                f"Cannot delete folder. Only empty folders are supported: {exc}"
            ) from exc
        return FileOperationResult(True, "delete_folder", path, path, 0, "deleted", not path.exists())

    @staticmethod
    def file_info(raw_path: str | Path) -> dict[str, object]:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileToolError("File does not exist")
        stat = path.stat()
        return {
            "path": str(path), "name": path.name, "extension": path.suffix.casefold(),
            "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns,
            "readable": os.access(path, os.R_OK), "writable": os.access(path, os.W_OK),
            "is_symlink": path.is_symlink(),
        }

    @staticmethod
    def folder_info(raw_path: str | Path, maximum_entries: int = 2000) -> dict[str, object]:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileToolError("Folder does not exist")
        files = folders = total_bytes = 0
        truncated = False
        try:
            for index, child in enumerate(path.iterdir()):
                if index >= max(1, min(10_000, maximum_entries)):
                    truncated = True
                    break
                if child.is_dir():
                    folders += 1
                else:
                    files += 1
                    try:
                        total_bytes += child.stat().st_size
                    except OSError:
                        pass
        except OSError as exc:
            raise FileToolError(f"Cannot inspect folder: {exc}") from exc
        return {
            "path": str(path), "name": path.name or str(path), "files": files,
            "folders": folders, "direct_file_bytes": total_bytes, "truncated": truncated,
            "readable": os.access(path, os.R_OK), "writable": os.access(path, os.W_OK),
        }

    def copy_file(self, raw_source: str | Path, raw_destination: str | Path) -> FileOperationResult:
        source, destination = self._operation_paths(raw_source, raw_destination)
        if source.stat().st_size > self.limits.max_file_bytes * 8:
            raise FileToolError("Source exceeds the bounded copy limit")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        except OSError as exc:
            raise FileToolError(f"Cannot copy file: {exc}") from exc
        return FileOperationResult(True, "copy", source, destination, destination.stat().st_size, "copied")

    def move_file(self, raw_source: str | Path, raw_destination: str | Path) -> FileOperationResult:
        source, destination = self._operation_paths(raw_source, raw_destination)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            moved = Path(shutil.move(str(source), str(destination))).resolve()
        except OSError as exc:
            raise FileToolError(f"Cannot move file: {exc}") from exc
        return FileOperationResult(True, "move", source, moved, moved.stat().st_size, "moved")

    def rename_file(self, raw_source: str | Path, new_name: str) -> FileOperationResult:
        source = Path(raw_source).expanduser().resolve()
        clean_name = Path(str(new_name).strip()).name
        if not source.is_file() or not clean_name or clean_name in {".", ".."}:
            raise FileToolError("A valid source file and new name are required")
        destination = source.with_name(clean_name)
        if destination.exists():
            raise FileToolError("Destination already exists; overwrite is not allowed")
        try:
            source.rename(destination)
        except OSError as exc:
            raise FileToolError(f"Cannot rename file: {exc}") from exc
        return FileOperationResult(True, "rename", source, destination, destination.stat().st_size, "renamed")

    @staticmethod
    def _operation_paths(
        raw_source: str | Path, raw_destination: str | Path,
    ) -> tuple[Path, Path]:
        source = Path(raw_source).expanduser().resolve()
        destination = Path(raw_destination).expanduser().resolve()
        if not source.is_file():
            raise FileToolError("Source file does not exist")
        if source == destination:
            raise FileToolError("Source and destination are identical")
        if destination.exists():
            raise FileToolError("Destination already exists; overwrite is not allowed")
        return source, destination

    @staticmethod
    def _decode_text(data: bytes) -> str:
        if not data:
            return ""
        encodings = ["utf-8-sig"]
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            encodings.insert(0, "utf-16")
        encodings.extend(["cp1256", "latin-1"])
        for encoding in encodings:
            try:
                text = data.decode(encoding)
                if "\x00" not in text or encoding == "utf-16":
                    return text
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def _inspect_text(
        self, path: Path, size: int, extension: str, mime_type: str
    ) -> FileInspection:
        limit = self.limits.max_text_preview_bytes
        try:
            with path.open("rb") as handle:
                data = handle.read(limit + 1)
        except OSError as exc:
            raise FileToolError(f"Cannot read text file: {exc}") from exc
        truncated = len(data) > limit or size > limit
        preview = self._decode_text(data[:limit]).replace("\x00", "")
        return FileInspection(
            path=path,
            name=path.name,
            extension=extension,
            mime_type=mime_type,
            size=size,
            kind="text",
            preview=preview,
            preview_truncated=truncated,
        )

    def _document_inspection(
        self, path: Path, size: int, mime_type: str, text: str,
    ) -> FileInspection:
        encoded = text.encode("utf-8")
        limit = self.limits.max_text_preview_bytes
        preview = encoded[:limit].decode("utf-8", errors="ignore")
        return FileInspection(
            path, path.name, path.suffix.casefold(), mime_type, size, "text",
            preview, len(encoded) > limit,
        )

    def _inspect_pdf(self, path: Path, size: int, mime_type: str) -> FileInspection:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FileToolError(
                "PDF text support needs the lightweight pypdf runtime dependency"
            ) from exc
        try:
            reader = PdfReader(path)
            pages: list[str] = []
            budget = self.limits.max_text_preview_bytes * 2
            for page in reader.pages[:40]:
                pages.append(page.extract_text() or "")
                if sum(len(value) for value in pages) >= budget:
                    break
        except Exception as exc:
            raise FileToolError(f"Cannot read PDF text: {exc}") from exc
        return self._document_inspection(path, size, mime_type, "\n\n".join(pages))

    def _inspect_docx(self, path: Path, size: int, mime_type: str) -> FileInspection:
        try:
            with zipfile.ZipFile(path) as archive:
                data = self._read_archive_member_bounded(
                    archive, "word/document.xml", self.limits.max_zip_total_preview_bytes
                )
            root = ElementTree.fromstring(data)
            text = "\n".join(
                element.text for element in root.iter() if element.tag.endswith("}t") and element.text
            )
        except FileToolError:
            raise
        except (OSError, KeyError, RuntimeError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise FileToolError(f"Cannot read DOCX text: {exc}") from exc
        return self._document_inspection(path, size, mime_type, text)

    def _inspect_xlsx(self, path: Path, size: int, mime_type: str) -> FileInspection:
        try:
            with zipfile.ZipFile(path) as archive:
                remaining = self.limits.max_zip_total_preview_bytes
                shared: list[str] = []
                if "xl/sharedStrings.xml" in archive.namelist():
                    data = self._read_archive_member_bounded(archive, "xl/sharedStrings.xml", remaining)
                    remaining -= len(data)
                    root = ElementTree.fromstring(data)
                    for item in list(root)[:10_000]:
                        shared.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")))
                lines: list[str] = []
                sheets = sorted(
                    name for name in archive.namelist()
                    if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
                )
                for sheet in sheets[:20]:
                    if remaining <= 0:
                        break
                    data = self._read_archive_member_bounded(archive, sheet, remaining)
                    remaining -= len(data)
                    root = ElementTree.fromstring(data)
                    values: list[str] = []
                    for index, cell in enumerate(node for node in root.iter() if node.tag.endswith("}c")):
                        if index >= 5_000:
                            break
                        value = next((node.text for node in cell if node.tag.endswith("}v")), "") or ""
                        if cell.attrib.get("t") == "s" and value.isdigit() and int(value) < len(shared):
                            value = shared[int(value)]
                        values.append(value)
                    lines.append(f"[{Path(sheet).stem}] " + " | ".join(values))
        except FileToolError:
            raise
        except (OSError, KeyError, RuntimeError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise FileToolError(f"Cannot read XLSX text: {exc}") from exc
        return self._document_inspection(path, size, mime_type, "\n".join(lines))

    def _read_archive_member_bounded(
        self, archive: zipfile.ZipFile, name: str, remaining_budget: int
    ) -> bytes:
        try:
            info = archive.getinfo(name)
        except KeyError as exc:
            raise FileToolError(f"Required archive member is missing: {name}") from exc
        limit = min(self.limits.max_zip_member_bytes, max(0, int(remaining_budget)))
        if limit <= 0 or info.file_size > limit:
            raise FileToolError(f"Archive member exceeds the safe preview budget: {name}")
        if info.flag_bits & 0x1:
            raise FileToolError(f"Encrypted archive member cannot be inspected: {name}")
        if info.compress_size and info.file_size > 100_000:
            if info.file_size / max(1, info.compress_size) > 200:
                raise FileToolError(f"Suspicious archive compression ratio: {name}")
        try:
            with archive.open(info, "r") as handle:
                data = handle.read(limit + 1)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise FileToolError(f"Cannot read archive member {name}: {exc}") from exc
        if len(data) > limit:
            raise FileToolError(f"Archive member exceeds the safe preview budget: {name}")
        return data

    @staticmethod
    def _safe_display_name(name: str) -> str:
        clean = name.replace("\\", "/").replace("\x00", "")
        parts = [part for part in PurePosixPath(clean).parts if part not in {".", "..", "/"}]
        return "/".join(parts) or "(unnamed)"

    def _member_preview(
        self,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        budget: int,
    ) -> tuple[str | None, bool, str, int]:
        if info.flag_bits & 0x1:
            return None, False, "encrypted; preview skipped", 0
        if info.file_size > self.limits.max_zip_member_bytes:
            return None, False, "text member exceeds preview limit", 0
        if info.compress_size and info.file_size > 100_000:
            if info.file_size / max(1, info.compress_size) > 200:
                return None, False, "suspicious compression ratio; preview skipped", 0
        allowed = min(info.file_size, self.limits.max_zip_member_bytes, budget)
        if allowed <= 0:
            return None, False, "total ZIP preview budget reached", 0
        try:
            with archive.open(info, "r") as handle:
                data = handle.read(allowed + 1)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            return None, False, f"preview unavailable: {exc}", 0
        truncated = len(data) > allowed or info.file_size > allowed
        consumed = min(len(data), allowed)
        return self._decode_text(data[:allowed]).replace("\x00", ""), truncated, "", consumed

    def _inspect_zip(self, path: Path, size: int, mime_type: str) -> FileInspection:
        try:
            with zipfile.ZipFile(path, "r") as archive:
                infos = archive.infolist()
                shown = infos[: self.limits.max_zip_entries]
                remaining_budget = self.limits.max_zip_total_preview_bytes
                previewed_members = 0
                entries: list[ZipEntry] = []
                for info in shown:
                    is_directory = info.is_dir()
                    preview: str | None = None
                    truncated = False
                    note = ""
                    extension = Path(info.filename).suffix.casefold()
                    if (
                        not is_directory
                        and extension in SUPPORTED_TEXT_EXTENSIONS
                        and previewed_members < 8
                    ):
                        preview, truncated, note, consumed = self._member_preview(
                            archive, info, remaining_budget
                        )
                        remaining_budget -= consumed
                        previewed_members += 1
                    entries.append(
                        ZipEntry(
                            name=self._safe_display_name(info.filename),
                            size=int(info.file_size),
                            compressed_size=int(info.compress_size),
                            is_directory=is_directory,
                            text_preview=preview,
                            preview_truncated=truncated,
                            note=note,
                        )
                    )
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            raise FileToolError(f"Invalid or unreadable ZIP file: {exc}") from exc

        return FileInspection(
            path=path,
            name=path.name,
            extension=".zip",
            mime_type=mime_type,
            size=size,
            kind="zip",
            zip_entries=tuple(entries),
            entries_truncated=len(infos) > len(entries),
        )

    def read_zip_text_member(self, raw_path: str | Path, member_name: str) -> str:
        path = Path(raw_path).expanduser().resolve()
        inspection = self.inspect(path)
        if inspection.kind != "zip":
            raise FileToolError("Selected file is not a ZIP archive")
        try:
            with zipfile.ZipFile(path, "r") as archive:
                info = archive.getinfo(member_name)
                if Path(info.filename).suffix.casefold() not in SUPPORTED_TEXT_EXTENSIONS:
                    raise FileToolError("ZIP member is not a supported text file")
                preview, _, note, _ = self._member_preview(
                    archive, info, self.limits.max_zip_member_bytes
                )
                if preview is None:
                    raise FileToolError(note or "ZIP member cannot be previewed")
                return preview
        except KeyError as exc:
            raise FileToolError("ZIP member was not found") from exc
