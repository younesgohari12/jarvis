from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.agent.intent_router import IntentRoute
from jarvis.agent.permissions import PermissionLayer, PermissionLevel
from jarvis.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]
    requires_confirmation: bool = False
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: str = "tool"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.call_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass(frozen=True, slots=True)
class PlanStep:
    index: int
    description: str
    tool_call: ToolCall | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"index": self.index, "description": self.description}
        if self.tool_call:
            value["tool_call"] = self.tool_call.to_dict()
        return value


@dataclass(frozen=True, slots=True)
class ActionPlan:
    original_request: str
    steps: tuple[PlanStep, ...]
    response_intent: str = ""
    needs_input: str = ""
    missing_tool: str = ""
    pending_arguments: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def actionable(self) -> bool:
        return any(step.tool_call for step in self.steps)

    @property
    def needs_confirmation(self) -> bool:
        return any(step.tool_call and step.tool_call.requires_confirmation for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "plan",
            "request": self.original_request,
            "steps": [step.to_dict() for step in self.steps],
            "response_intent": self.response_intent,
            "needs_input": self.needs_input,
            "missing_tool": self.missing_tool,
            "pending_arguments": self.pending_arguments,
            "dry_run": self.dry_run,
        }


class ToolPlanner:
    """Capability-aware planner. Incomplete calls become clarification plans."""

    _TOOLS = {
        "open_url": "open_url",
        "open_app": "open_app",
        "close_app": "close_app",
        "close_default_browser": "close_default_browser",
        "focus_app": "focus_app",
        "is_app_running": "is_app_running",
        "restart_app": "restart_app",
        "minimize_app": "minimize_app",
        "maximize_app": "maximize_app",
        "restore_app": "restore_app",
        "find_installed_app": "find_installed_app",
        "list_installed_apps": "list_installed_apps",
        "list_running_apps": "list_running_apps",
        "open_default_browser": "open_default_browser",
        "web_search": "web_search",
        "youtube_search": "web_search",
        "read_file": "file_inspect",
        "zip_inspect": "file_inspect",
        "list_folder": "list_folder",
        "find_file": "find_file",
        "find_folder": "find_folder",
        "find_and_open_folder": "find_and_open_folder",
        "close_apps": "close_apps",
        "close_all_browsers": "close_all_browsers",
        "write_file": "write_file",
        "copy_file": "copy_file",
        "move_file": "move_file",
        "rename_file": "rename_file",
        "create_folder": "create_folder",
        "create_file": "create_file",
        "append_file": "append_file",
        "delete_file": "delete_file",
        "delete_folder": "delete_folder",
        "open_file": "open_file",
        "file_info": "file_info",
        "folder_info": "folder_info",
        "enumerate_windows": "enumerate_windows",
        "find_window": "find_window",
        "focus_window": "focus_window",
        "close_window": "close_window",
        "minimize_window": "minimize_window",
        "maximize_window": "maximize_window",
        "restore_window": "restore_window",
        "move_window": "move_window",
        "resize_window": "resize_window",
        "browser_open_url": "browser_open_url",
        "search_google": "search_google",
        "search_youtube": "search_youtube",
        "current_url": "current_url",
        "navigate_back": "navigate_back",
        "navigate_forward": "navigate_forward",
        "refresh_browser": "refresh_browser",
        "new_tab": "new_tab",
        "close_tab": "close_tab",
        "switch_tab": "switch_tab",
        "browser_read_page": "browser_read_page",
        "browser_links": "browser_links",
        "browser_open_link": "browser_open_link",
        "browser_click": "browser_click",
        "browser_fill": "browser_fill",
        "type_text": "type_text",
        "press_key": "press_key",
        "hotkey": "hotkey",
        "click": "click",
        "double_click": "double_click",
        "right_click": "right_click",
        "scroll": "scroll",
        "move_mouse": "move_mouse",
        "clipboard_clear": "clipboard_clear",
        "run_command": "run_command",
        "volume_up": "volume_up",
        "volume_down": "volume_down",
        "set_volume": "set_volume",
        "mute": "mute",
        "unmute": "unmute",
        "set_brightness": "set_brightness",
        "cpu_info": "cpu_info",
        "gpu_info": "gpu_info",
        "ram_info": "ram_info",
        "network_info": "network_info",
        "battery_info": "battery_info",
        "process_info": "process_info",
        "shutdown_system": "shutdown_system",
        "restart_system": "restart_system",
        "sleep_system": "sleep_system",
        "logoff_system": "logoff_system",
        "disk_info": "disk_info",
        "clipboard_read": "clipboard_read",
        "clipboard_write": "clipboard_write",
        "web_research": "web_research",
        "open_folder": "open_folder",
        "system_info": "system_info",
        "calculator": "calculator",
        "date_time": "date_time",
        # v0.2 compatibility routes:
        "file_read": "file_inspect",
        "zip_list": "file_inspect",
        "internet_search": "web_search",
        "open_browser": "open_default_browser",
        "open_settings": "ui_open_settings",
        "clear_conversation": "ui_clear_conversation",
        "undo_last_action": "undo_last_action",
        "diagnose_system": "diagnose_system",
        "forget_memory": "forget_memory",
        "list_skills": "list_skills",
        "list_plugins": "list_plugins",
    }

    _RESPONSE_INTENTS = {
        "dangerous_request": "tool_confirm_dangerous",
        "clear_conversation": "tool_confirm_clear",
    }

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry

    def _select_tool(self, route: IntentRoute) -> str:
        # A routed intent is already the most specific capability decision.  Prefer
        # its explicit mapping before the semantic registry fallback; otherwise an
        # "open folder by name" request can be reduced to the generic open_folder
        # tool and lose the bounded search step.
        mapped = self._TOOLS.get(route.intent, "")
        if mapped and (self.registry is None or self.registry.spec(mapped) is not None):
            return mapped
        if self.registry and route.action and route.entity_type:
            selected = self.registry.select(route.action, route.entity_type)
            if selected:
                return selected.name
        return mapped

    def plan(
        self,
        text: str,
        route: IntentRoute,
        attachment_path: Path | None = None,
    ) -> ActionPlan | None:
        if route.route_type != "tool":
            return None
        if route.intent == "multi_step_task":
            return None
        if route.intent == "file_selection_workflow":
            raw = route.arguments.get("workflow", {})
            workflow = dict(raw) if isinstance(raw, dict) else {}
            final_action = str(workflow.get("final_action", ""))
            if final_action not in {"move_file", "copy_file", "open_file", "delete_file"}:
                return ActionPlan(
                    text, (), "tool_clarification", "action", "file_workflow",
                    workflow, route.dry_run,
                )
            search_arguments = {
                key: workflow.get(key)
                for key in ("root", "extension", "modified", "order_by", "descending", "limit")
            }
            search_call = ToolCall("search_files", search_arguments)
            selected_path: dict[str, str] = {"$ref": "step1.items.0.path"}
            if final_action in {"move_file", "copy_file"}:
                final_arguments: dict[str, Any] = {
                    "source": selected_path,
                    "destination": str(workflow.get("destination", "")),
                }
            else:
                final_arguments = {"path": selected_path}
            final_call = ToolCall(
                final_action,
                final_arguments,
                requires_confirmation=final_action in {"move_file", "copy_file", "delete_file"},
            )
            return ActionPlan(
                text,
                (
                    PlanStep(1, "Search with the requested metadata filters", search_call),
                    PlanStep(2, "Select the ranked result without guessing"),
                    PlanStep(3, f"Execute {final_action} with the selected file", final_call),
                ),
                "tool_file_selection_workflow_result",
                dry_run=route.dry_run,
            )
        if route.intent == "dangerous_request":
            call = ToolCall(
                "blocked_dangerous_request", dict(route.arguments), requires_confirmation=True
            )
            return ActionPlan(
                text, (PlanStep(1, "Require explicit confirmation for a dangerous action", call),),
                "tool_confirm_dangerous",
            )

        tool = self._select_tool(route)
        if not tool:
            return None
        arguments = dict(route.arguments)
        if tool == "file_inspect" and not arguments.get("path"):
            if attachment_path is None:
                return ActionPlan(text, (), "tool_need_file", "attachment")
            arguments["path"] = str(attachment_path)
        if self.registry:
            missing = self.registry.validate(tool, arguments)
            if missing:
                return ActionPlan(
                    text,
                    (PlanStep(1, f"Resolve missing arguments for {tool}"),),
                    "tool_clarification",
                    missing[0],
                    tool,
                    arguments,
                )
        confirmation = bool(route.requires_confirmation)
        if self.registry:
            specification = self.registry.spec(tool)
            if specification:
                category = specification.effective_permission_category(arguments)
                level = PermissionLayer.level_for(category)
                confirmation = confirmation or bool(
                    level is not None and level >= PermissionLevel.MODIFY
                )
        call = ToolCall(tool, arguments, confirmation)
        return ActionPlan(
            text,
            (
                PlanStep(1, f"Resolve {route.intent} arguments"),
                PlanStep(2, f"Execute safe tool {tool}", call),
                PlanStep(3, "Store tool result in task context"),
            ),
            self._RESPONSE_INTENTS.get(route.intent, f"tool_{route.intent}_result"),
            dry_run=route.dry_run,
        )

    @staticmethod
    def combine(original_request: str, plans: list[ActionPlan]) -> ActionPlan:
        for plan in plans:
            if plan.needs_input:
                return ActionPlan(
                    original_request, (), "tool_clarification",
                    plan.needs_input, plan.missing_tool, plan.pending_arguments,
                )
        steps: list[PlanStep] = []
        for plan in plans:
            for step in plan.steps:
                if step.tool_call:
                    steps.append(
                        PlanStep(len(steps) + 1, step.description, step.tool_call)
                    )
        return ActionPlan(
            original_request, tuple(steps), "tool_multi_step_result",
            dry_run=any(plan.dry_run for plan in plans),
        )
