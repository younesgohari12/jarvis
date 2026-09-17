from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.agent.dialogue_subject import DialogueSubjectResolver


class TaskContext:
    """Entity-based task state with bounded recency stacks."""

    BROWSERS = {"chrome", "edge", "firefox", "default", "default_browser"}
    LIMIT = 5

    @staticmethod
    def _entity(entity_type: str, entity_id: str, label: str = "") -> dict[str, str]:
        return {"type": entity_type, "id": entity_id, "label": label or entity_id}

    @classmethod
    def _push(cls, state: dict[str, Any], key: str, value: Any) -> None:
        values = list(state.get(key, [])) if isinstance(state.get(key), list) else []
        values = [item for item in values if item != value]
        values.insert(0, value)
        state[key] = values[: cls.LIMIT]

    @staticmethod
    def _succeeded(result: Any) -> bool:
        if result is None:
            return True
        success = getattr(result, "success", None)
        return bool(success) if success is not None else result is not False

    @classmethod
    def update(
        cls,
        current: dict[str, Any],
        tool: str,
        arguments: dict[str, Any],
        result: Any = None,
    ) -> dict[str, Any]:
        state = dict(current)
        if not cls._succeeded(result):
            return state
        state["last_tool"] = tool
        state["last_action"] = tool
        cls._push(state, "recent_tools", tool)
        entity: dict[str, str] | None = None
        if tool in {
            "open_app", "close_app", "focus_app", "restart_app", "is_app_running",
            "minimize_app", "maximize_app", "restore_app", "close_default_browser",
        }:
            app_id = str(
                arguments.get("app", "") or getattr(result, "app_id", "")
            ).strip()
            if app_id:
                label = str(getattr(result, "label", "") or app_id)
                entity = cls._entity("app", app_id, label)
                cls._push(state, "recent_apps", entity)
                if tool == "open_app":
                    state["last_opened_app"] = entity
                    state["last_active_app"] = entity
                elif tool in {"close_app", "close_default_browser"}:
                    state["last_closed_app"] = entity
                else:
                    state["last_active_app"] = entity
                if app_id in cls.BROWSERS:
                    state["last_browser"] = entity
        elif tool in {
            "open_url", "web_search", "web_research", "search_google", "search_youtube",
            "browser_open_url", "navigate_back", "navigate_forward", "refresh_browser",
        }:
            url = str(arguments.get("url", "")).strip()
            website = str(arguments.get("website", "")).strip()
            browser = str(arguments.get("browser") or arguments.get("target_browser") or "").strip()
            if url:
                state["last_opened_url"] = url
                cls._push(state, "recent_urls", url)
            if website:
                entity = cls._entity("website", website, website)
                state["last_website"] = entity
            elif url:
                entity = cls._entity("url", url, url)
            if browser:
                state["last_browser"] = cls._entity("app", browser, browser)
            if tool in {"web_search", "web_research"}:
                query = str(arguments.get("query", "")).strip()
                if query:
                    state["last_search"] = query
                    cls._push(state, "recent_searches", query)
                    if tool == "web_research":
                        state["last_research_query"] = query
                subject = str(arguments.get("subject", "")).strip()
                if not subject and tool == "web_research":
                    subject = DialogueSubjectResolver.extract_subject(query)
                if subject:
                    state["last_information_subject"] = subject
                    if tool == "web_research":
                        state["last_research_subject"] = subject
                    cls._push(state, "recent_information_subjects", subject)
            elif tool in {"search_google", "search_youtube"}:
                query = str(arguments.get("query", "")).strip()
                if query:
                    state["last_search"] = query
                    cls._push(state, "recent_searches", query)
        elif tool in {
            "open_folder", "open_drive", "list_folder", "find_file", "find_folder",
            "find_and_open_folder", "create_folder", "delete_folder", "folder_info",
        }:
            path = str(arguments.get("path") or arguments.get("folder") or "").strip()
            if not path:
                path = str(arguments.get("root") or "").strip()
            if path:
                drive = path[:1].upper() if len(path) >= 3 and path[1:3] in {":\\", ":/"} else ""
                entity_type = "drive" if drive and path.rstrip("\\/") == f"{drive}:" else "folder"
                entity_id = drive if entity_type == "drive" else path
                entity = cls._entity(entity_type, entity_id, path)
                state["last_folder"] = path
                if drive:
                    state["last_drive"] = cls._entity("drive", drive, f"Drive {drive}")
                cls._push(state, "recent_folders", entity)
        elif tool in {
            "file_inspect", "open_file", "create_file", "write_file", "append_file",
            "rename_file", "copy_file", "move_file", "delete_file", "file_info",
        }:
            path = str(arguments.get("path", "")).strip()
            if not path:
                path = str(arguments.get("destination") or arguments.get("source") or "").strip()
            if path:
                entity = cls._entity("file", path, Path(path).name)
                state["last_file"] = path
                state["last_folder"] = str(Path(path).expanduser().parent)
                cls._push(state, "recent_files", entity)
        elif tool in {
            "focus_window", "close_window", "minimize_window", "maximize_window",
            "restore_window", "move_window", "resize_window",
        }:
            window = getattr(result, "window", None)
            title = str(getattr(window, "title", "") or arguments.get("window", "")).strip()
            handle = str(getattr(window, "handle", "") or title)
            if handle:
                entity = cls._entity("window", handle, title or handle)
                state["last_window"] = entity
                cls._push(state, "recent_windows", entity)
        if entity:
            state["last_entity"] = entity
            cls._push(state, "recent_entities", entity)
        cls._push(
            state,
            "recent_tasks",
            {"tool": tool, "arguments": dict(arguments)},
        )
        return state
