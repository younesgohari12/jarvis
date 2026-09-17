from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.agent.context import ContextEngine
from jarvis.agent.core import JarvisAgent
from jarvis.agent.permissions import PermissionLayer
from jarvis.brain.model import BrainLoadError, HybridNeuralBrain
from jarvis.brain.responses import ResponseEngine
from jarvis.config import AppConfig, load_config
from jarvis.entities.resolver import EntityResolver
from jarvis.knowledge.store import KnowledgeStore
from jarvis.knowledge.rag import RAGStore
from jarvis.learning.failures import FailureCollector
from jarvis.logging_config import configure_logging
from jarvis.memory.store import MemoryStore
from jarvis.personalities.engine import PersonalityEngine
from jarvis.plugins.registry import PluginRegistry
from jarvis.runtime.hardware import HardwareManager
from jarvis.runtime.diagnostics import DiagnosticManager
from jarvis.runtime.app_index import AppIndex
from jarvis.research.engine import ResearchEngine
from jarvis.search.engine import SearchEngine
from jarvis.skills.registry import SkillRegistry
from jarvis.tools.browser import BrowserManager
from jarvis.tools.browser_automation import BrowserAutomation, BrowserDOMResult
from jarvis.tools.calculator import CalculatorTool
from jarvis.tools.clipboard import ClipboardTool
from jarvis.tools.datetime_tool import DateTimeTool
from jarvis.tools.desktop import DesktopTool
from jarvis.tools.files import FileManager
from jarvis.tools.internet import InternetTool
from jarvis.tools.input_control import InputController
from jarvis.tools.processes import ProcessManager
from jarvis.tools.registry import ToolRegistry, ToolSpec
from jarvis.tools.system import SystemInfoTool
from jarvis.tools.system_control import SystemController
from jarvis.tools.terminal import CommandRunner
from jarvis.tools.windows import WindowManager


@dataclass(slots=True)
class RuntimeContext:
    config: AppConfig
    memory: MemoryStore
    hardware: HardwareManager
    brain: HybridNeuralBrain | None
    neural: Any | None
    knowledge: KnowledgeStore
    rag: RAGStore
    tools: ToolRegistry
    personalities: PersonalityEngine
    entities: EntityResolver
    app_index: AppIndex
    processes: ProcessManager
    browser: BrowserManager
    browser_automation: BrowserAutomation
    windows: WindowManager
    failures: FailureCollector
    skills: SkillRegistry
    plugins: PluginRegistry
    agent: JarvisAgent
    startup_warnings: tuple[str, ...]
    logger: logging.Logger

    def close(self) -> None:
        self.browser_automation.close()
        if self.neural is not None:
            self.neural.close()
        self.knowledge.close()
        self.rag.close()
        self.memory.close()


def _required(arguments: dict[str, Any], key: str) -> Any:
    value = arguments.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing tool argument: {key}")
    return value


def _initialize_defaults(config: AppConfig, memory: MemoryStore) -> None:
    defaults: dict[str, str | bool] = {
        "personality": config.default_personality,
        "performance": config.gui.performance,
        "internet_enabled": config.internet.enabled_by_default,
        "memory_enabled": True,
        "animation": config.gui.animation,
        "timestamps": config.gui.timestamps,
        "theme": config.gui.theme,
        "language": config.gui.language,
        "sampling_profile": "Balanced",
    }
    for key, value in defaults.items():
        if memory.get_setting(key) is None:
            memory.set_setting(key, value)


def build_runtime(root: Path, *, debug: bool = False) -> RuntimeContext:
    # Startup: Config -> Hardware -> Memory -> Personality -> Brains -> Tools -> Agent.
    config = load_config(root)
    cancel_event = threading.Event()
    logger = configure_logging(config.paths.log_file, debug=debug)
    logger.info("Starting Jarvis %s (debug=%s)", config.version, debug)
    hardware = HardwareManager()
    memory = MemoryStore(
        config.paths.database,
        config.memory.max_history_rows,
        config.memory.max_sessions,
        config.memory.retention_days,
    )
    _initialize_defaults(config, memory)
    selected = memory.get_setting("personality", config.default_personality)
    personalities = PersonalityEngine(
        config.paths.personalities, selected or config.default_personality
    )

    warnings: list[str] = []
    brain: HybridNeuralBrain | None
    brain_error = ""
    try:
        brain = HybridNeuralBrain.load(config.paths.model, config.brain)
        logger.info(
            "Hybrid brain loaded: %s parameters, %s bytes",
            brain.parameter_count, config.paths.model.stat().st_size,
        )
    except BrainLoadError as exc:
        brain = None
        brain_error = str(exc)
        warnings.append(f"Brain load error: {exc}")
        logger.error("Brain load error: %s", exc)

    neural: Any | None = None
    neural_error = ""
    try:
        from jarvis.neural.conversation import NeuralConversationEngine

        neural = NeuralConversationEngine(config.paths.neural_model, config.paths.tokenizer)
        logger.info(
            "JARVIS Smart Brain registered lazily: %s parameters, dataset=%s, pretrained_source=%s",
            neural.parameter_count,
            neural.metadata.get("dataset_version"),
            neural.metadata.get("pretrained_source"),
        )
    except Exception as exc:
        neural_error = str(exc)
        warnings.append(f"Neural language model load error: {exc}")
        logger.error("Neural language model load error: %s", exc)

    knowledge = KnowledgeStore(
        config.paths.knowledge_database,
        config.paths.knowledge_seed,
        config.knowledge.seed_version,
        config.knowledge.max_documents,
    )
    rag = RAGStore(config.paths.rag_database, config.knowledge.max_documents)
    entities = EntityResolver(config.paths.websites, config.paths.apps)
    app_index = AppIndex(config.paths.app_index, entities.apps)
    permissions = PermissionLayer()
    tools = ToolRegistry(permissions)
    files = FileManager(config.files)
    internet = InternetTool(
        config.internet.timeout_seconds,
        config.internet.max_response_bytes,
        config.internet.max_search_results,
        config.internet.max_redirects,
        cancel_event,
    )
    search = SearchEngine(internet, config.internet.fetch_top_pages)
    research = ResearchEngine(
        search, config.paths.logs_dir / "search.jsonl",
        lambda: memory.get_bool_setting("memory_enabled", True),
    )
    desktop = DesktopTool()
    windows = WindowManager()
    processes = ProcessManager(app_index, windows)
    browser = BrowserManager(processes)
    browser_automation = BrowserAutomation(browser)
    calculator = CalculatorTool()
    clipboard = ClipboardTool()
    input_control = InputController()
    command_runner = CommandRunner(cancel_event=cancel_event)
    system_control = SystemController()
    failures = FailureCollector(config.paths.failure_queue)
    date_time = DateTimeTool()
    system_info = SystemInfoTool(hardware)
    diagnostics = DiagnosticManager(
        hardware, memory, tools,
        (
            config.paths.model, config.paths.neural_model, config.paths.tokenizer,
            config.paths.root / "models" / "bilingual_fluency_v11.json",
            config.paths.root / "models" / "cognitive_skills_v11.json",
        ),
    )

    def undo_last_action(_arguments: dict[str, Any]) -> dict[str, Any]:
        record = memory.last_reversible_action(agent.session_id)
        if record is None:
            return {"status": "nothing_to_undo"}
        result = tools.invoke(
            record.inverse_tool, record.inverse_arguments, confirmed=True
        )
        success = getattr(result, "success", result is not False and result is not None)
        if not success:
            return {
                "status": "undo_failed", "action_id": record.id,
                "inverse_tool": record.inverse_tool,
            }
        memory.mark_action_undone(record.id)
        return {
            "status": "undone", "action_id": record.id,
            "original_tool": record.tool, "inverse_tool": record.inverse_tool,
            "result": result,
        }

    def browser_open_link(arguments: dict[str, Any]) -> BrowserDOMResult:
        links_result = browser_automation.links(100)
        if not links_result.success:
            return links_result
        links = list(links_result.data.get("links", []))
        index = int(arguments.get("index", 0))
        if index < 0 or index >= len(links):
            return BrowserDOMResult(False, "open_link", True, "invalid_link_index")
        url = str(links[index].get("url", ""))
        return browser_automation.open_url(url, automation=True)

    def find_and_open_folder(arguments: dict[str, Any]) -> dict[str, Any]:
        matches = files.find_folder(
            _required(arguments, "root"), str(_required(arguments, "name"))
        )
        if not matches:
            return {"status": "not_found", "matches": [], "opened": ""}
        if len(matches) > 1:
            return {
                "status": "ambiguous",
                "matches": [str(path) for path in matches],
                "opened": "",
            }
        opened = desktop.open_folder(matches[0])
        return {"status": "opened", "matches": [str(matches[0])], "opened": opened}

    tools.register(ToolSpec(
        "file_inspect", "Read supported text or inspect ZIP without extraction", "filesystem_read",
        lambda args: files.inspect(_required(args, "path")),
        required=("path",), semantic_actions=("read",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "knowledge_index", "Index a supported local document in the lexical RAG store", "filesystem_read",
        lambda args: rag.index_file(files, _required(args, "path")), required=("path",),
    ))
    tools.register(ToolSpec(
        "knowledge_search", "Retrieve grounded chunks from local indexed documents", "information",
        lambda args: rag.search(str(_required(args, "query"))), required=("query",),
    ))
    tools.register(ToolSpec(
        "zip_read_text", "Read a supported text member from a ZIP", "filesystem_read",
        lambda args: files.read_zip_text_member(_required(args, "path"), _required(args, "member")),
        required=("path", "member"), semantic_actions=("read",), entity_types=("zip_member",),
    ))
    tools.register(ToolSpec(
        "list_folder", "List a bounded number of folder entries", "filesystem_read",
        lambda args: files.list_folder(_required(args, "path")),
        required=("path",), semantic_actions=("list",), entity_types=("folder", "drive"),
    ))
    tools.register(ToolSpec(
        "find_file", "Find files within a bounded folder traversal", "filesystem_read",
        lambda args: files.find_file(_required(args, "folder"), _required(args, "query")),
        required=("folder", "query"), semantic_actions=("find",), entity_types=("file", "folder"),
    ))
    tools.register(ToolSpec(
        "search_files", "Filter and rank files for a multi-step plan", "filesystem_read",
        lambda args: files.search_files(
            _required(args, "root"),
            extension=str(args.get("extension", "")),
            modified=str(args.get("modified", "any")),
            order_by=str(args.get("order_by", "modified")),
            descending=bool(args.get("descending", True)),
            limit=int(args.get("limit", 20)),
        ),
        required=("root",),
        optional=("extension", "modified", "order_by", "descending", "limit"),
        semantic_actions=("find", "search"), entity_types=("file", "folder"),
    ))
    tools.register(ToolSpec(
        "find_folder", "Find folders within a bounded traversal", "filesystem_read",
        lambda args: files.find_folder(_required(args, "root"), _required(args, "name")),
        required=("root", "name"), semantic_actions=("find",), entity_types=("folder",),
    ))
    tools.register(ToolSpec(
        "find_and_open_folder", "Find a unique folder and open it", "external_app",
        find_and_open_folder, required=("root", "name"),
        semantic_actions=("open",), entity_types=("folder",),
    ))
    tools.register(ToolSpec(
        "write_file", "Write bounded UTF-8 text after confirmation", "write_file",
        lambda args: files.write_text(_required(args, "path"), str(_required(args, "content"))),
        required=("path", "content"), risk="confirm",
        semantic_actions=("write",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "copy_file", "Copy one validated file after confirmation", "write_file",
        lambda args: files.copy_file(_required(args, "source"), _required(args, "destination")),
        required=("source", "destination"), risk="confirm",
        semantic_actions=("copy",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "move_file", "Move one validated file after confirmation", "move",
        lambda args: files.move_file(_required(args, "source"), _required(args, "destination")),
        required=("source", "destination"), risk="confirm",
        semantic_actions=("move",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "rename_file", "Rename one validated file after confirmation", "rename",
        lambda args: files.rename_file(_required(args, "source"), str(_required(args, "new_name"))),
        required=("source", "new_name"), risk="confirm",
        semantic_actions=("rename",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "internet_check", "Check internet connectivity", "network",
        lambda _args: internet.check_connection(),
    ))
    tools.register(ToolSpec(
        "internet_fetch_text", "Fetch bounded public HTTP/HTTPS text", "network",
        lambda args: internet.fetch_text(str(_required(args, "url"))),
        required=("url",),
    ))
    tools.register(ToolSpec(
        "web_search", "Search the web or open a search in a selected browser", "network",
        lambda args: (
            browser.open_url(
                str(_required(args, "url")),
                str(args.get("target_browser", "")),
            )
            if args.get("url")
            else browser.search(
                str(_required(args, "query")), str(args.get("target_browser", ""))
            )
            if args.get("target_browser")
            else search.search(
                str(_required(args, "query")), str(args.get("subject", ""))
            )
        ),
        required=("query",), optional=("target_browser", "url", "website", "subject"),
        semantic_actions=("search",), entity_types=("information", "browser", "website"),
    ))
    tools.register(ToolSpec(
        "web_research", "Run bounded multi-query research and rank sources", "network",
        lambda args: research.research(
            str(_required(args, "query")), str(args.get("subject", ""))
        ),
        required=("query",), optional=("subject",),
        semantic_actions=("research",), entity_types=("information",),
    ))
    # Compatibility alias used by v0.2 extensions.
    tools.register(ToolSpec(
        "internet_search", "Return raw public text search results", "network",
        lambda args: internet.search(str(_required(args, "query"))),
        required=("query",),
    ))
    tools.register(ToolSpec(
        "open_url", "Open a validated URL in the default or selected browser", "external_app",
        lambda args: browser.open_url(
            str(_required(args, "url")), str(args.get("browser", ""))
        ),
        required=("url",), optional=("browser", "website"),
        semantic_actions=("open",), entity_types=("url", "website"),
    ))
    tools.register(ToolSpec(
        "browser_open", "Compatibility alias for opening a URL", "external_app",
        lambda args: browser.open_url(str(_required(args, "url"))),
        required=("url",),
    ))
    tools.register(ToolSpec(
        "open_default_browser", "Open the system default browser", "external_app",
        lambda _args: browser.open_default(),
        semantic_actions=("open",), entity_types=("browser",),
    ))
    tools.register(ToolSpec(
        "open_browser", "Compatibility capability for opening the default browser", "external_app",
        lambda _args: browser.open_default(),
    ))
    tools.register(ToolSpec(
        "open_app", "Open a configured or dynamically discovered application", "external_app",
        lambda args: processes.open_app(str(_required(args, "app"))),
        required=("app",), semantic_actions=("open",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "close_app", "Close a resolved running desktop application", "process_terminate",
        lambda args: processes.close_app(str(_required(args, "app"))),
        required=("app",), semantic_actions=("close",), entity_types=("app", "browser"),
    ))
    tools.register(ToolSpec(
        "close_default_browser", "Close the actual Windows default browser", "process_terminate",
        lambda _args: processes.close_default_browser(),
        semantic_actions=("close",), entity_types=("browser",),
    ))
    tools.register(ToolSpec(
        "close_apps", "Close several resolved applications", "process_terminate",
        lambda args: processes.close_apps(tuple(str(value) for value in _required(args, "apps"))),
        required=("apps",), semantic_actions=("close",), entity_types=("app_collection",),
    ))
    tools.register(ToolSpec(
        "close_all_browsers", "Close all supported browser process trees", "process_terminate",
        lambda _args: processes.close_all_browsers(),
        semantic_actions=("close",), entity_types=("browser_collection",),
    ))
    tools.register(ToolSpec(
        "close_browser", "Close a resolved browser process", "process_terminate",
        lambda args: processes.close_app(str(_required(args, "app"))),
        required=("app",),
    ))
    tools.register(ToolSpec(
        "focus_app", "Focus a resolved running desktop application", "external_app",
        lambda args: processes.focus_app(str(_required(args, "app"))),
        required=("app",), semantic_actions=("focus",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "restart_app", "Restart a resolved desktop application", "process_terminate",
        lambda args: processes.restart_app(str(_required(args, "app"))),
        required=("app",), semantic_actions=("restart",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "is_app_running", "Check whether a resolved application is running", "information",
        lambda args: processes.is_app_running(str(_required(args, "app"))),
        required=("app",), semantic_actions=("is_running",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "find_running_app", "Find running processes belonging to an application", "information",
        lambda args: processes.find_running_app(str(_required(args, "app"))),
        required=("app",),
    ))
    tools.register(ToolSpec(
        "list_running_apps", "List running processes without changing them", "information",
        lambda _args: processes.list_running_apps(),
        semantic_actions=("list",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "open_folder", "Open an existing folder in the platform file manager", "external_app",
        lambda args: desktop.open_folder(_required(args, "path")),
        required=("path",), semantic_actions=("open",), entity_types=("folder", "drive"),
    ))
    tools.register(ToolSpec(
        "calculator", "Evaluate bounded arithmetic without eval", "information",
        lambda args: calculator.calculate(str(_required(args, "expression"))),
        required=("expression",),
    ))
    tools.register(ToolSpec(
        "date_time", "Return local date, time and weekday", "information",
        lambda _args: date_time.current(),
    ))
    tools.register(ToolSpec(
        "system_info", "Return lightweight read-only hardware information", "information",
        lambda _args: system_info.get_info(),
    ))
    tools.register(ToolSpec(
        "disk_info", "Return disk capacity and free-space information", "information",
        lambda args: system_info.disk_info(str(args.get("path", ""))), optional=("path",),
    ))
    tools.register(ToolSpec(
        "clipboard_read", "Read text from the system clipboard", "clipboard_access",
        lambda _args: clipboard.read_text(), semantic_actions=("read",), entity_types=("clipboard",),
    ))
    tools.register(ToolSpec(
        "clipboard_write", "Write bounded text to the clipboard after confirmation", "write_file",
        lambda args: clipboard.write_text(str(_required(args, "text"))), required=("text",),
        risk="confirm", semantic_actions=("write",), entity_types=("clipboard",),
    ))
    tools.register(ToolSpec(
        "clipboard_clear", "Clear the text clipboard", "clipboard_destructive",
        lambda _args: clipboard.clear(), semantic_actions=("delete",), entity_types=("clipboard",),
    ))

    # Dynamic application index and verified Win32 window controls.
    tools.register(ToolSpec(
        "find_installed_app", "Search configured, Start Menu, Registry and Program Files app index", "information",
        lambda args: app_index.find(str(_required(args, "query"))), required=("query",),
        semantic_actions=("find",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "list_installed_apps", "List the bounded installed-program index", "information",
        lambda args: app_index.all_apps(refresh=bool(args.get("refresh", False))), optional=("refresh",),
        semantic_actions=("list",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "refresh_app_index", "Refresh the Windows installed-program cache", "information",
        lambda _args: app_index.refresh(),
    ))
    tools.register(ToolSpec(
        "minimize_app", "Minimize a running application's real window", "external_app",
        lambda args: processes.minimize_app(str(_required(args, "app"))), required=("app",),
        semantic_actions=("minimize",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "maximize_app", "Maximize a running application's real window", "external_app",
        lambda args: processes.maximize_app(str(_required(args, "app"))), required=("app",),
        semantic_actions=("maximize",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "restore_app", "Restore a running application's real window", "external_app",
        lambda args: processes.restore_app(str(_required(args, "app"))), required=("app",),
        semantic_actions=("restore",), entity_types=("app",),
    ))
    tools.register(ToolSpec(
        "enumerate_windows", "Enumerate visible top-level Windows windows", "information",
        lambda _args: windows.enumerate_windows(), semantic_actions=("list",), entity_types=("window",),
    ))
    tools.register(ToolSpec(
        "find_window", "Find visible windows by fuzzy title", "information",
        lambda args: windows.find_window(str(_required(args, "window"))), required=("window",),
        semantic_actions=("find",), entity_types=("window",),
    ))
    tools.register(ToolSpec(
        "focus_window", "Focus a visible Windows window", "external_app",
        lambda args: windows.focus(str(_required(args, "window"))), required=("window",),
        semantic_actions=("focus",), entity_types=("window",),
    ))
    tools.register(ToolSpec(
        "close_window", "Request graceful close for a visible Windows window", "process_terminate",
        lambda args: windows.close(str(_required(args, "window"))), required=("window",),
        semantic_actions=("close",), entity_types=("window",),
    ))
    for tool_name, method, action in (
        ("minimize_window", windows.minimize, "minimize"),
        ("maximize_window", windows.maximize, "maximize"),
        ("restore_window", windows.restore, "restore"),
    ):
        tools.register(ToolSpec(
            tool_name, f"{action.title()} a visible Windows window", "external_app",
            lambda args, handler=method: handler(str(_required(args, "window"))), required=("window",),
            semantic_actions=(action,), entity_types=("window",),
        ))
    tools.register(ToolSpec(
        "move_window", "Move a real window to screen coordinates", "caution",
        lambda args: windows.move(
            str(_required(args, "window")), int(_required(args, "x")), int(_required(args, "y"))
        ), required=("window", "x", "y"), semantic_actions=("move",), entity_types=("window",),
    ))
    tools.register(ToolSpec(
        "resize_window", "Resize a real window", "caution",
        lambda args: windows.resize(
            str(_required(args, "window")), int(_required(args, "width")), int(_required(args, "height"))
        ), required=("window", "width", "height"), semantic_actions=("resize",), entity_types=("window",),
    ))

    # Complete bounded file-system capabilities. Destructive tools are always confirmed.
    tools.register(ToolSpec(
        "open_drive", "Open an existing drive root", "external_app",
        lambda args: desktop.open_folder(_required(args, "path")), required=("path",),
        semantic_actions=("open",), entity_types=("drive",),
    ))
    tools.register(ToolSpec(
        "open_file", "Open an existing file with its registered Windows application", "external_app",
        lambda args: desktop.open_file(_required(args, "path")), required=("path",),
        semantic_actions=("open",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "create_folder", "Create one folder tree", "write_file",
        lambda args: files.create_folder(_required(args, "path")), required=("path",),
        semantic_actions=("create",), entity_types=("folder",),
    ))
    tools.register(ToolSpec(
        "create_file", "Create one bounded UTF-8 text file", "write_file",
        lambda args: files.create_file(_required(args, "path"), str(args.get("content", ""))),
        required=("path",), optional=("content",), semantic_actions=("create",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "append_file", "Append bounded UTF-8 text after confirmation", "write_file",
        lambda args: files.append_text(_required(args, "path"), str(_required(args, "content"))),
        required=("path", "content"), risk="confirm", semantic_actions=("append",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "truncate_file", "Restore a file to a previously recorded size", "write_file",
        lambda args: files.truncate_file(_required(args, "path"), int(_required(args, "size"))),
        required=("path", "size"), risk="confirm",
    ))
    tools.register(ToolSpec(
        "delete_file", "Delete one explicit file after dangerous confirmation", "delete",
        lambda args: files.delete_file(_required(args, "path")), required=("path",), risk="dangerous",
        semantic_actions=("delete",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "delete_folder", "Delete one explicit empty folder after dangerous confirmation", "delete",
        lambda args: files.delete_folder(_required(args, "path")), required=("path",), risk="dangerous",
        semantic_actions=("delete",), entity_types=("folder",),
    ))
    tools.register(ToolSpec(
        "file_info", "Return read-only file metadata", "filesystem_read",
        lambda args: files.file_info(_required(args, "path")), required=("path",),
        semantic_actions=("inspect",), entity_types=("file",),
    ))
    tools.register(ToolSpec(
        "folder_info", "Return bounded read-only folder metadata", "filesystem_read",
        lambda args: files.folder_info(_required(args, "path")), required=("path",),
        semantic_actions=("inspect",), entity_types=("folder",),
    ))
    tools.register(ToolSpec(
        "undo_last_action", "Run the verified inverse of the latest reversible action",
        "sensitive_action", undo_last_action,
        risk="confirm", semantic_actions=("undo",), entity_types=("action_history",),
    ))
    tools.register(ToolSpec(
        "forget_memory", "Forget an explicitly selected memory key", "sensitive_action",
        lambda args: {
            "status": "forgotten",
            "removed": memory.forget(
                str(_required(args, "scope")), str(args.get("key", ""))
            ),
        },
        required=("scope",), optional=("key",), risk="confirm",
        semantic_actions=("forget",), entity_types=("memory",),
    ))
    tools.register(ToolSpec(
        "diagnose_system", "Run read-only Jarvis health and recovery checks", "information",
        lambda args: diagnostics.run(str(args.get("scope", "jarvis"))),
        optional=("scope",), semantic_actions=("diagnose",), entity_types=("system",),
    ))

    # Browser navigation works with ordinary installed browsers; DOM tools use
    # the optional lightweight automation environment and never embed Chromium.
    tools.register(ToolSpec(
        "browser_open_url", "Open a URL in ordinary or DOM-automation mode", "external_app",
        lambda args: browser_automation.open_url(
            str(_required(args, "url")), str(args.get("browser", "")), automation=bool(args.get("automation", False))
        ), required=("url",), optional=("browser", "automation"),
        semantic_actions=("open",), entity_types=("url",),
    ))
    tools.register(ToolSpec(
        "search_google", "Search Google in the selected browser", "network",
        lambda args: browser_automation.search_google(
            str(_required(args, "query")), str(args.get("browser", "")), automation=bool(args.get("automation", False))
        ), required=("query",), optional=("browser", "automation"),
        semantic_actions=("search",), entity_types=("browser", "information"),
    ))
    tools.register(ToolSpec(
        "search_youtube", "Search YouTube in the selected browser", "network",
        lambda args: browser_automation.search_youtube(
            str(_required(args, "query")), str(args.get("browser", "")), automation=bool(args.get("automation", False))
        ), required=("query",), optional=("browser", "automation"),
        semantic_actions=("search",), entity_types=("website",),
    ))
    tools.register(ToolSpec("current_url", "Return the last known or automated current URL", "information", lambda _args: browser_automation.current_url()))
    tools.register(ToolSpec("navigate_back", "Navigate automated browser back", "external_app", lambda _args: browser_automation.navigate_back()))
    tools.register(ToolSpec("navigate_forward", "Navigate automated browser forward", "external_app", lambda _args: browser_automation.navigate_forward()))
    tools.register(ToolSpec("refresh_browser", "Refresh automated browser page", "external_app", lambda _args: browser_automation.refresh()))
    tools.register(ToolSpec(
        "new_tab", "Create a browser automation tab", "external_app",
        lambda args: browser_automation.new_tab(str(args.get("url", ""))), optional=("url",),
    ))
    tools.register(ToolSpec(
        "close_tab", "Close a browser automation tab", "external_app",
        lambda args: browser_automation.close_tab(int(args["index"]) if "index" in args else None), optional=("index",),
    ))
    tools.register(ToolSpec(
        "switch_tab", "Switch browser automation tab", "external_app",
        lambda args: browser_automation.switch_tab(int(_required(args, "index"))), required=("index",),
    ))
    tools.register(ToolSpec("browser_read_page", "Read bounded visible page text through DOM", "network", lambda _args: browser_automation.read_page()))
    tools.register(ToolSpec("browser_links", "List bounded page links through DOM", "network", lambda _args: browser_automation.links()))
    tools.register(ToolSpec(
        "browser_open_link", "Open a numbered link from the current DOM page", "external_app",
        browser_open_link, required=("index",), semantic_actions=("click",), entity_types=("web_link",),
    ))
    tools.register(ToolSpec(
        "browser_click", "Click a DOM element; sensitive labels require elevated confirmation", "browser_interaction",
        lambda args: browser_automation.click(
            str(args.get("label", "")), str(args.get("selector", "")), sensitive=bool(args.get("sensitive", False))
        ), optional=("label", "selector", "sensitive"), risk_resolver=browser_automation.click_risk,
        semantic_actions=("click",), entity_types=("web_element",),
    ))
    tools.register(ToolSpec(
        "browser_fill", "Fill a simple DOM input without submitting it", "browser_interaction",
        lambda args: browser_automation.fill(
            str(_required(args, "field")), str(_required(args, "text")), selector=str(args.get("selector", ""))
        ), required=("field", "text"), optional=("selector",),
        semantic_actions=("type",), entity_types=("web_element",),
    ))

    # Keyboard/mouse are explicit fallback capabilities, never the first planner choice.
    tools.register(ToolSpec("type_text", "Type Unicode text with Win32 SendInput fallback", "ui_control", lambda args: input_control.type_text(str(_required(args, "text"))), required=("text",)))
    tools.register(ToolSpec("press_key", "Press a named keyboard key", "ui_control", lambda args: input_control.press_key(str(_required(args, "key")), int(args.get("presses", 1))), required=("key",), optional=("presses",)))
    tools.register(ToolSpec("hotkey", "Press a bounded key combination", "ui_control", lambda args: input_control.hotkey(tuple(_required(args, "keys"))), required=("keys",)))
    tools.register(ToolSpec("click", "Click the current mouse position", "ui_control", lambda _args: input_control.click("left", 1)))
    tools.register(ToolSpec("double_click", "Double-click the current mouse position", "ui_control", lambda _args: input_control.click("left", 2)))
    tools.register(ToolSpec("right_click", "Right-click the current mouse position", "ui_control", lambda _args: input_control.click("right", 1)))
    tools.register(ToolSpec("scroll", "Scroll the mouse wheel", "ui_control", lambda args: input_control.scroll(int(_required(args, "amount"))), required=("amount",)))
    tools.register(ToolSpec("move_mouse", "Move the mouse within current screen bounds", "ui_control", lambda args: input_control.move_mouse(int(_required(args, "x")), int(_required(args, "y"))), required=("x", "y")))

    # Deny-by-default terminal and Windows system control.
    tools.register(ToolSpec(
        "run_command", "Run one policy-classified command without a shell", "caution",
        lambda args: command_runner.run(
            str(_required(args, "command")), cwd=str(args.get("cwd", "")),
            timeout_seconds=float(args.get("timeout_seconds", 10.0)),
        ), required=("command",), optional=("cwd", "timeout_seconds"),
        risk_resolver=command_runner.risk_category,
        semantic_actions=("run",), entity_types=("command",),
    ))
    tools.register(ToolSpec("volume_up", "Raise Windows master volume", "caution", lambda args: system_control.volume_up(int(args.get("steps", 1))), optional=("steps",)))
    tools.register(ToolSpec("volume_down", "Lower Windows master volume", "caution", lambda args: system_control.volume_down(int(args.get("steps", 1))), optional=("steps",)))
    tools.register(ToolSpec("set_volume", "Set Windows master volume percentage", "caution", lambda args: system_control.set_volume(int(_required(args, "percent"))), required=("percent",)))
    tools.register(ToolSpec("mute", "Mute Windows audio", "caution", lambda _args: system_control.mute()))
    tools.register(ToolSpec("unmute", "Restore Windows audio", "caution", lambda _args: system_control.unmute()))
    tools.register(ToolSpec("set_brightness", "Set supported display brightness", "caution", lambda args: system_control.set_brightness(int(_required(args, "percent"))), required=("percent",)))
    tools.register(ToolSpec("battery_info", "Return Windows battery information", "information", lambda _args: system_control.battery_info()))
    tools.register(ToolSpec("network_info", "Return bounded local network information", "information", lambda _args: system_control.network_info()))
    tools.register(ToolSpec("process_info", "Return running process information", "information", lambda _args: processes.list_running_apps()))
    tools.register(ToolSpec("cpu_info", "Return CPU information", "information", lambda _args: {key: value for key, value in system_info.get_info().items() if key in {"cpu", "physical_cores", "logical_cores", "platform", "release"}}))
    tools.register(ToolSpec("gpu_info", "Return GPU information without fabricating availability", "information", lambda _args: {key: value for key, value in system_info.get_info().items() if key in {"gpu", "gpu_available", "platform"}}))
    tools.register(ToolSpec("ram_info", "Return RAM information", "information", lambda _args: {key: value for key, value in system_info.get_info().items() if key in {"ram_total_mb", "platform"}}))
    tools.register(ToolSpec("shutdown_system", "Schedule Windows shutdown after confirmation", "power_control", lambda args: system_control.shutdown(int(args.get("delay_seconds", 15))), optional=("delay_seconds",), risk="dangerous"))
    tools.register(ToolSpec("restart_system", "Schedule Windows restart after confirmation", "power_control", lambda args: system_control.restart(int(args.get("delay_seconds", 15))), optional=("delay_seconds",), risk="dangerous"))
    tools.register(ToolSpec("sleep_system", "Suspend Windows after confirmation", "power_control", lambda _args: system_control.sleep(), risk="dangerous"))
    tools.register(ToolSpec("logoff_system", "Log off Windows after confirmation", "power_control", lambda _args: system_control.logoff(), risk="dangerous"))
    tools.register(ToolSpec(
        "ui_open_settings", "Ask the desktop UI to open settings", "information",
        lambda _args: {"event": "open_settings"},
    ))
    tools.register(ToolSpec(
        "ui_clear_conversation", "Clear the visible conversation after confirmation", "information",
        lambda _args: {"event": "clear_conversation"}, risk="confirm",
    ))

    skills = SkillRegistry(config.paths.root / "skills", tools)
    skills.discover()
    tools.register(ToolSpec(
        "list_skills", "List validated local Jarvis skills and capability health", "information",
        lambda _args: skills.report(), semantic_actions=("list",), entity_types=("skill",),
    ))
    plugins = PluginRegistry(config.paths.root / "plugins", tools, skills)
    plugins.discover()
    tools.register(ToolSpec(
        "list_plugins", "List validated local capability plugins", "information",
        lambda _args: plugins.report(), semantic_actions=("list",), entity_types=("plugin",),
    ))

    responses = ResponseEngine(config.paths.responses, personalities)
    context_engine = ContextEngine(config.paths.context_rules)
    agent = JarvisAgent(
        config, memory, knowledge, context_engine, personalities, brain,
        responses, tools, entities, logger, brain_error, debug, neural, rag, failures,
        cancel_event,
    )
    return RuntimeContext(
        config, memory, hardware, brain, neural, knowledge, rag, tools, personalities,
        entities, app_index, processes, browser, browser_automation, windows, failures,
        skills, plugins, agent, tuple(warnings), logger,
    )
