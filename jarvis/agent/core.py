from __future__ import annotations

import logging
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from jarvis.agent.context import ContextEngine
from jarvis.agent.constraints_v16 import ConstraintAwareComposer, ConstraintExtractor, ResponseConstraints, ResponseVerifier
from jarvis.agent.conversation_chain import ConversationChainReasoner
from jarvis.agent.dialogue_subject import ContextualQuery, DialogueSubjectResolver
from jarvis.agent.executor import ToolExecutor
from jarvis.agent.bilingual_fluency import BilingualFluencyEngine
from jarvis.agent.intent_router import IntentRoute, IntentRouter
from jarvis.agent.learning import LearningParser
from jarvis.agent.planner import ActionPlan, PlanStep, ToolCall, ToolPlanner
from jarvis.agent.reasoning import ReasoningEngine
from jarvis.agent.rewrite_v18 import RewriteEngineV18
from jarvis.agent.recovery import FailureRecovery
from jarvis.agent.task_context import TaskContext
from jarvis.brain.model import HybridNeuralBrain
from jarvis.brain.responses import ResponseEngine
from jarvis.config import AppConfig
from jarvis.entities.resolver import EntityResolver
from jarvis.knowledge.store import KnowledgeStore
from jarvis.knowledge.rag import RAGStore
from jarvis.learning.failures import FailureCollector
from jarvis.memory.store import MemoryStore, Message
from jarvis.personalities.engine import PersonalityEngine
from jarvis.research.engine import ResearchReport
from jarvis.nlu.action_parser import ActionParser
from jarvis.search.engine import SearchReport
from jarvis.tools.calculator import CalculationResult
from jarvis.tools.browser import BrowserActionResult
from jarvis.tools.datetime_tool import DateTimeResult
from jarvis.tools.drives import DriveResolver, KnownFolderResolver
from jarvis.tools.files import FileInspection, FileOperationResult, FileToolError, FolderListing
from jarvis.tools.processes import ProcessResult
from jarvis.tools.registry import ToolError, ToolRegistry
from jarvis.telemetry import StructuredLogs
from jarvis.utils.text import detect_language, normalize_text


StatusCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class AgentReply:
    text: str
    intent: str
    confidence: float
    language: str
    status: str = "READY"
    event: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PendingInput:
    tool: str
    missing: str
    arguments: dict[str, Any]
    original_request: str


class JarvisAgent:
    def __init__(
        self,
        config: AppConfig,
        memory: MemoryStore,
        knowledge: KnowledgeStore,
        context_engine: ContextEngine,
        personalities: PersonalityEngine,
        brain: HybridNeuralBrain | None,
        responses: ResponseEngine,
        tools: ToolRegistry,
        entities: EntityResolver,
        logger: logging.Logger | None = None,
        brain_error: str = "",
        debug: bool = False,
        neural: Any | None = None,
        rag: RAGStore | None = None,
        failures: FailureCollector | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.knowledge = knowledge
        self.context_engine = context_engine
        self.personalities = personalities
        self.brain = brain
        self.responses = responses
        self.tools = tools
        self.entities = entities
        self.logger = logger or logging.getLogger("jarvis.agent")
        self.brain_error = brain_error
        self.debug = debug
        self.neural = neural
        self.rag = rag
        self.failures = failures
        self._cancel_event = cancel_event or threading.Event()
        self.events = StructuredLogs(config.paths.logs_dir)
        self.router = IntentRouter(
            brain,
            config.paths.intent_signatures,
            entities,
            memory,
            config.paths.root / "models" / "cognitive_skills_v11.json",
            config.paths.root / "models" / "semantic_router_v18.npz",
        )
        self.planner = ToolPlanner(tools)
        self.executor = ToolExecutor(tools)
        self.reasoning = ReasoningEngine(knowledge, responses, config.knowledge.minimum_score)
        self.response_verifier = ResponseVerifier()
        self.rewrite_engine = RewriteEngineV18()
        self._active_constraints = ResponseConstraints()
        self.fluency = BilingualFluencyEngine(
            config.paths.root / "models" / "bilingual_fluency_v11.json"
        )
        self.dialogue_subjects = DialogueSubjectResolver()
        self.conversation_chain = ConversationChainReasoner(
            config.paths.root / "models" / "dialogue_followup_v10.json"
        )
        self.learning = LearningParser(entities)
        self.session_id, self.resumed_session = memory.start_session(
            config.memory.resume_session_hours
        )
        self.memory.cleanup()
        self._short_term: deque[tuple[str, str, dict[str, Any]]] = deque(
            maxlen=config.memory.short_term_messages
        )
        for message in memory.recent_messages(self.session_id, config.memory.short_term_messages):
            self._short_term.append((message.role, message.content, message.metadata))
        self._attachment: FileInspection | None = None
        self._pending_plan: ActionPlan | None = None
        self._pending_input: PendingInput | None = None
        self._last_language = "fa"
        self._last_assistant_message_id: int | None = None
        self._status_callback: StatusCallback | None = None
        self._lock = threading.RLock()

    @property
    def current_personality(self) -> str:
        return self.personalities.current.name

    @property
    def current_attachment(self) -> FileInspection | None:
        return self._attachment

    @property
    def memory_enabled(self) -> bool:
        return self.memory.get_bool_setting("memory_enabled", True)

    @property
    def internet_enabled(self) -> bool:
        return self.memory.get_bool_setting(
            "internet_enabled", self.config.internet.enabled_by_default
        )

    def set_status_callback(self, callback: StatusCallback | None) -> None:
        self._status_callback = callback

    def cancel_current(self) -> None:
        """Request cooperative cancellation without waiting for the agent lock."""
        self._cancel_event.set()

    def _cancelled_reply(self, route: IntentRoute, language: str, *, tool_finished: bool = False) -> AgentReply:
        self._pending_plan = None
        self._pending_input = None
        if language == "en":
            text = "Cancelled. The current tool may already have completed." if tool_finished else "Cancelled."
        else:
            text = "لغو شد؛ ممکن است ابزار جاری کارش را تمام کرده باشد." if tool_finished else "لغو شد."
        return self._record_reply(text, "cancelled", route, language)

    def _emit_status(self, status: str) -> None:
        callback = self._status_callback
        if callback:
            try:
                callback(status)
            except Exception:
                self.logger.debug("Status callback failed", exc_info=True)

    def recent_history(self) -> list[Message]:
        return self.memory.recent_messages(self.session_id, self.config.memory.recent_messages)

    def get_setting(self, key: str, default: str = "") -> str:
        return self.memory.get_setting(key, default) or default

    def update_setting(self, key: str, value: str | bool) -> None:
        allowed = {
            "performance", "internet_enabled", "memory_enabled", "animation",
            "timestamps", "theme", "language",
        }
        if key not in allowed:
            raise ValueError(f"Unsupported user setting: {key}")
        self.memory.set_setting(key, value)

    def set_personality(self, name: str) -> str:
        with self._lock:
            profile = self.personalities.select(name)
            self.memory.set_setting("personality", profile.name)
            return (
                f"Personality switched to {profile.display_name}."
                if self._last_language == "en"
                else f"حالت شخصیت روی {profile.display_name} تنظیم شد."
            )

    def add_feedback(self, positive: bool, note: str = "") -> int:
        return self.memory.add_feedback(
            self.session_id, self._last_assistant_message_id, positive, note,
            self.config.learning.max_feedback_rows,
        )

    def index_attachment(self) -> dict[str, object]:
        if self._attachment is None:
            raise FileToolError("Attach a supported document first")
        if self.rag is None:
            raise FileToolError("Knowledge index is unavailable")
        return self.rag.index_inspection(self._attachment)

    def _remember_message(self, role: str, content: str, metadata: dict[str, Any]) -> int | None:
        self._short_term.append((role, content, metadata))
        message_id: int | None = None
        if self.memory_enabled:
            self.events.chat.write(
                "message", session_id=self.session_id, role=role,
                content=content[:4000], metadata=metadata,
            )
            message_id = self.memory.add_message(self.session_id, role, content, metadata)
            if role == 'user':
                from jarvis.memory.context_v21 import MemoryContextV21
                MemoryContextV21(self.memory).observe(content,self.session_id)
        if role == "assistant":
            self._last_assistant_message_id = message_id
        return message_id

    def attach_file(self, path: str | Path, language: str | None = None) -> AgentReply:
        chosen_language = language or self._last_language
        self._emit_status("READING")
        try:
            inspection = self.tools.invoke("file_inspect", {"path": str(path)})
            if not isinstance(inspection, FileInspection):
                raise FileToolError("File tool returned an invalid result")
            self._attachment = inspection
            if self.rag is not None and inspection.kind == "text" and inspection.preview.strip():
                try:
                    self.rag.index_inspection(inspection)
                except Exception:
                    self.logger.debug("Attachment RAG indexing failed", exc_info=True)
            if self.memory_enabled:
                self.memory.remember_attachment(
                    self.session_id, inspection.name, str(inspection.path),
                    inspection.kind, inspection.size,
                )
                state = self.memory.get_context(self.session_id)
                topic = str(state.pop("topic", "file")) or "file"
                state.update({"last_file": str(inspection.path), "last_folder": str(inspection.path.parent)})
                self.memory.set_context(self.session_id, topic, state)
            text = inspection.display(chosen_language)
            metadata = {
                "intent": "file_attached", "file_name": inspection.name,
                "file_size": inspection.size, "file_kind": inspection.kind,
            }
            self._remember_message("assistant", text, metadata)
            return AgentReply(
                text, "file_attached", 1.0, chosen_language, event="attachment",
                data={"name": inspection.name, "size": inspection.size, "kind": inspection.kind},
            )
        except (ToolError, FileToolError) as exc:
            self.logger.warning("File attachment failed: %s", exc)
            text = (
                f"I couldn't inspect that file: {exc}"
                if chosen_language == "en" else f"نتونستم فایل را بررسی کنم: {exc}"
            )
            return AgentReply(text, "file_error", 1.0, chosen_language, status="ERROR")
        finally:
            self._emit_status("READY")

    def _slots(self, route: IntentRoute) -> dict[str, str]:
        slots = dict(route.slots)
        user_name = self.memory.get_fact("user_name") if self.memory_enabled else None
        if user_name:
            slots["user_name"] = user_name
        now = datetime.now().astimezone()
        slots["time"] = now.strftime("%H:%M")
        slots["date"] = now.strftime("%Y-%m-%d")
        return slots

    def _render(
        self, intent: str, language: str, message: str,
        slots: dict[str, Any] | None = None,
    ) -> str:
        return self.responses.render(
            intent, language, message, self.memory.turn_count(self.session_id), slots
        )

    def _record_decision(self, original: str, route: IntentRoute, tool: str = "") -> None:
        entity = ",".join(route.entities)
        if self.memory_enabled:
            self.events.agent.write(
                "decision", session_id=self.session_id, input=original[:1000],
                intent=route.intent, route_type=route.route_type, action=route.action,
                entity=entity, arguments=route.arguments, confidence=round(route.confidence, 5),
                tool=tool, confirmation=route.requires_confirmation, source=route.source,
            )
        if self.debug:
            self.logger.info(
                "DECISION input=%r intent=%s entity=%s confidence=%.3f tool=%s confirmation=%s source=%s",
                original[:120] if self.memory_enabled else "[memory disabled]",
                route.intent, entity if self.memory_enabled else "", route.confidence, tool,
                route.requires_confirmation, route.source,
            )
            if self.memory_enabled:
                self.memory.add_decision_log(
                    self.session_id, original, route.intent, entity, route.confidence,
                    tool, route.requires_confirmation, route.source,
                )

    def _record_reply(
        self,
        text: str,
        intent: str,
        route: IntentRoute,
        language: str,
        *,
        event: str = "",
        data: dict[str, Any] | None = None,
        status: str = "READY",
    ) -> AgentReply:
        if self._active_constraints.active and route.route_type != "tool":
            repaired = self.response_verifier.repair(text, self._active_constraints)
            text = repaired.text
            if data is None:
                data = {}
            data = dict(data)
            data["constraint_verification"] = {
                "valid": repaired.valid,
                "violations": list(repaired.violations),
            }
            # v20 independently checks the final, repaired user-visible text.
            c=self._active_constraints
            checks=[{'kind':'required','value':x} for x in c.include_terms]
            checks += [{'kind':'forbidden','value':x} for x in c.exclude_terms]
            for kind,value in [('sentence_count',c.exact_sentences),('max_words',c.max_words),('language',c.language_only)]:
                if value is not None and value!='': checks.append({'kind':kind,'value':value})
            independent=self.reasoning.local_intelligence.verifier.verify_text(text,checks)
            data['constraint_verification_v20']=independent.to_dict()
        metadata = {
            "intent": intent,
            "confidence": round(route.confidence, 5),
            "route_source": route.source,
            "route_type": route.route_type,
            "unknown_reason": route.unknown_reason,
            "personality": self.current_personality,
            "brain_available": self.brain is not None,
            "decision_mode": route.decision_mode,
        }
        self._remember_message("assistant", text, metadata)
        self._remember_answer_context(text, intent)
        self._emit_status("READY" if status != "ERROR" else "ERROR")
        return AgentReply(
            text, intent, route.confidence, language, status=status,
            event=event, data=data or {},
        )

    def _remember_answer_context(self, text: str, intent: str) -> None:
        """Keep only a recent substantive answer for bounded follow-ups.

        This is deliberately session context rather than a global fact: phrases
        such as "summarize that" must never attach to an old, unrelated answer.
        """
        if not self.memory_enabled:
            return
        excluded = {
            "empty", "cancelled", "context_clarification", "fact_remembered",
            "fact_not_found", "greeting", "greeting_named", "smalltalk",
            "ask_user_name_known", "ask_user_name_unknown", "tool_failure",
            "tool_cancelled", "internet_disabled", "learning_clarification",
        }
        if intent in excluded or intent.startswith(("tool_open_", "tool_close_")):
            return
        clean = re.sub(r"\s+", " ", str(text)).strip()
        if len(clean) < 16:
            return
        state = self.memory.get_context(self.session_id)
        topic = str(state.pop("topic", "conversation")) or "conversation"
        state["last_assistant_text"] = clean[:4000]
        state["last_answer_turn"] = self.memory.turn_count(self.session_id)
        state["last_answer_intent"] = intent
        self.memory.set_context(self.session_id, topic, state)

    def _task_context(self) -> dict[str, Any]:
        if not self.memory_enabled:
            return {"topic": ""}
        return self.memory.get_context(self.session_id)

    def _annotate_information_route(
        self,
        route: IntentRoute,
        contextual: ContextualQuery,
    ) -> IntentRoute:
        if route.intent not in {"web_search", "web_research"}:
            return route
        arguments = dict(route.arguments)
        subject = contextual.subject
        if not subject and route.intent == "web_research":
            subject = self.dialogue_subjects.extract_subject(
                str(arguments.get("query", ""))
            )
        if subject:
            arguments["subject"] = subject
        if contextual.used_context:
            arguments["query"] = contextual.query
        return replace(route, arguments=arguments, referenced=contextual.used_context)

    def _remember_information_mode(
        self, contextual: ContextualQuery, route: IntentRoute,
    ) -> None:
        if (
            not self.memory_enabled
            or not contextual.mode
            or not contextual.subject
            or (
                route.route_type not in {"information", "think", "unknown", "tool"}
                and route.intent not in {"knowledge_question", "factual_question", "web_search"}
            )
        ):
            return
        current = self.memory.get_context(self.session_id)
        state = dict(current)
        state.pop("topic", None)
        state["last_information_subject"] = contextual.subject
        state["last_information_mode"] = contextual.mode
        state["last_information_turn"] = self.memory.turn_count(self.session_id)
        self.memory.set_context(self.session_id, "information", state)

    def _update_task_context(self, call: ToolCall, result: Any) -> None:
        if not self.memory_enabled:
            return
        current = self.memory.get_context(self.session_id)
        topic = str(current.pop("topic", "task")) or "task"
        updated = TaskContext.update(current, call.tool, call.arguments, result)
        if updated.get("last_information_subject"):
            updated["last_information_turn"] = self.memory.turn_count(self.session_id)
        self.memory.set_context(self.session_id, topic, updated)
        for key in (
            "last_opened_app", "last_closed_app", "last_active_app", "last_opened_url",
            "last_folder", "last_file", "last_drive", "last_search", "last_window",
            "last_command", "last_tool", "last_information_subject", "last_research_query",
        ):
            value = updated.get(key)
            if isinstance(value, dict):
                value = value.get("id", "")
            if isinstance(value, str) and value:
                self.memory.set_fact(key, value)

    def _tool_success_text(
        self, call: ToolCall, result: Any, language: str, original: str,
    ) -> tuple[str, str]:
        def styled(base: str) -> str:
            tone = self.personalities.response_override(
                "tool_success", language, original
            )
            if tone:
                return f"{base} {tone}".strip()
            return self.personalities.apply_style(
                base, f"tool_{call.tool}", language, original
            )

        if call.tool == "open_url":
            website = str(call.arguments.get("website", ""))
            url = str(call.arguments.get("url", ""))
            if not website:
                match = self.entities.resolve_website(url)
                website = match.entity_id if match else "URL"
            label = self.entities.website(website).name if self.entities.website(website) else website
            if isinstance(result, BrowserActionResult) and result.label not in {"", "Browser"}:
                label = label or result.label
            base = f"Opened {label}." if language == "en" else f"{label} رو باز کردم."
            return styled(base), "tool_open_url_result"
        if call.tool == "open_app":
            app_id = str(call.arguments.get("app", ""))
            app = self.entities.app(app_id)
            label = result.label if isinstance(result, ProcessResult) else app.name if app else app_id
            if language == "fa" and app_id == "chrome":
                label = "کروم"
            base = f"Opened {label}." if language == "en" else f"{label} رو باز کردم."
            return styled(base), "tool_open_app_result"
        if call.tool == "close_default_browser" and isinstance(result, ProcessResult):
            label = result.label or (
                "the default browser" if language == "en" else "مرورگر پیش‌فرض"
            )
            if result.code == "background_only":
                base = (
                    f"{label} had no open window; only its background service was running."
                    if language == "en" else
                    f"پنجره‌ای از {label} باز نبود؛ فقط پردازش پس‌زمینه‌اش فعال بود."
                )
            else:
                base = (
                    f"Closed the default browser ({label})."
                    if language == "en" else
                    f"مرورگر پیش‌فرض ({label}) رو بستم."
                )
            return styled(base), "tool_close_default_browser_result"
        if call.tool in {"close_app", "focus_app", "restart_app"}:
            app_id = str(call.arguments.get("app", ""))
            app = self.entities.app(app_id)
            label = result.label if isinstance(result, ProcessResult) else app.name if app else app_id
            if call.tool == "close_app":
                if isinstance(result, ProcessResult) and result.code == "already_stopped":
                    base = f"{label} was already closed." if language == "en" else f"{label} از قبل بسته بود."
                else:
                    base = f"Closed {label}." if language == "en" else f"{label} رو بستم."
                return styled(base), "tool_close_app_result"
            if call.tool == "focus_app":
                base = f"Focused {label}." if language == "en" else f"{label} رو آوردم جلو."
                return styled(base), "tool_focus_app_result"
            base = f"Restarted {label}." if language == "en" else f"{label} رو دوباره اجرا کردم."
            return styled(base), "tool_restart_app_result"
        if call.tool in {"close_apps", "close_all_browsers"} and isinstance(result, tuple):
            closed = [value.label for value in result if isinstance(value, ProcessResult) and value.success]
            failed = [value.label for value in result if isinstance(value, ProcessResult) and not value.success]
            if language == "en":
                base = f"Browser close check finished. Closed/already stopped: {', '.join(closed) or 'none'}."
                if failed:
                    base += f" Could not close: {', '.join(failed)}."
            else:
                base = f"بررسی مرورگرها تمام شد؛ بسته یا از قبل متوقف: {('، '.join(closed) or 'هیچ‌کدام')}."
                if failed:
                    base += f" بسته نشد: {'، '.join(failed)}."
            return styled(base), "tool_close_apps_result"
        if call.tool == "is_app_running":
            app_id = str(call.arguments.get("app", ""))
            app = self.entities.app(app_id)
            label = app.name if app else app_id
            running = bool(result)
            if language == "en":
                return f"{label} is {'running' if running else 'not running'}.", "tool_app_status_result"
            return f"{label} {'در حال اجراست' if running else 'در حال اجرا نیست'}.", "tool_app_status_result"
        if call.tool == "open_default_browser":
            base = "Opened the default browser." if language == "en" else "مرورگر پیش‌فرض رو باز کردم."
            return styled(base), "tool_browser_result"
        if call.tool == "file_inspect" and isinstance(result, FileInspection):
            return result.display(language), "tool_file_result"
        if call.tool == "list_folder" and isinstance(result, FolderListing):
            return result.display(language), "tool_folder_result"
        if call.tool == "find_file" and isinstance(result, tuple):
            if result:
                return "\n".join(str(value) for value in result), "tool_find_result"
            return ("No matching file was found." if language == "en" else "فایل مطابقی پیدا نشد."), "tool_find_result"
        if call.tool == "find_folder" and isinstance(result, tuple):
            if result:
                return "\n".join(str(value) for value in result), "tool_find_folder_result"
            return (
                "No matching folder was found." if language == "en" else "پوشهٔ مطابقی پیدا نشد."
            ), "tool_find_folder_result"
        if call.tool == "find_and_open_folder" and isinstance(result, dict):
            status = str(result.get("status", ""))
            matches = [str(value) for value in result.get("matches", [])]
            if status == "opened":
                opened = str(result.get("opened", matches[0] if matches else ""))
                return (
                    f"Found and opened: {opened}" if language == "en" else f"پیداش کردم و بازش کردم: {opened}"
                ), "tool_folder_opened"
            if status == "ambiguous":
                listing = "\n".join(f"{index}. {value}" for index, value in enumerate(matches, 1))
                return (
                    f"I found several folders. Which one?\n{listing}"
                    if language == "en"
                    else f"چند پوشه پیدا کردم؛ کدوم رو باز کنم؟\n{listing}"
                ), "tool_folder_choice"
            return (
                "I couldn't find that folder." if language == "en" else "اون پوشه رو پیدا نکردم."
            ), "tool_find_folder_result"
        if call.tool == "open_folder":
            path = str(call.arguments.get("path", ""))
            if re.fullmatch(r"[A-Za-z]:[\\/]", path):
                drive = path[0].upper()
                return (
                    f"Opened drive {drive}." if language == "en" else f"درایو {drive} رو باز کردم."
                ), "tool_folder_opened"
            return (f"Opened folder: {result}" if language == "en" else f"پوشه را باز کردم: {result}"), "tool_folder_opened"
        if call.tool == "calculator" and isinstance(result, CalculationResult):
            return f"{result.expression} = {result.value}", "tool_calculation_result"
        if call.tool == "date_time" and isinstance(result, DateTimeResult):
            if language == "en":
                return f"{result.weekday_en}, {result.date} — {result.time}", "tool_datetime_result"
            return f"{result.weekday_fa}، {result.date} — ساعت {result.time}", "tool_datetime_result"
        if call.tool == "system_info" and isinstance(result, dict):
            disk = int(result.get("disk_free_mb", 0) or 0)
            if language == "en":
                return (
                    f"CPU: {result.get('cpu')}\nRAM: {result.get('ram_total_mb')} MB\n"
                    f"Cores: {result.get('physical_cores')} physical / {result.get('logical_cores')} logical\n"
                    f"64-bit: {'yes' if result.get('is_64_bit') else 'no'}\nFree disk: {disk} MB\n"
                    f"GPU: {result.get('gpu')}\nPlatform: {result.get('platform')} {result.get('release')}"
                ), "tool_system_result"
            return (
                f"CPU: {result.get('cpu')}\nRAM: {result.get('ram_total_mb')} مگابایت\n"
                f"هسته‌ها: {result.get('physical_cores')} فیزیکی / {result.get('logical_cores')} منطقی\n"
                f"۶۴ بیتی: {'بله' if result.get('is_64_bit') else 'خیر'}\nفضای خالی: {disk} مگابایت\n"
                f"GPU: {result.get('gpu')}\nسیستم‌عامل: {result.get('platform')} {result.get('release')}"
            ), "tool_system_result"
        if call.tool == "disk_info" and isinstance(result, dict):
            if not result.get("available"):
                return (
                    "Disk information is unavailable." if language == "en" else "اطلاعات دیسک در دسترس نیست."
                ), "tool_disk_result"
            free_gb = int(result.get("free_bytes", 0)) / (1024 ** 3)
            total_gb = int(result.get("total_bytes", 0)) / (1024 ** 3)
            return (
                f"Disk {result.get('path')}: {free_gb:.1f} GB free of {total_gb:.1f} GB."
                if language == "en"
                else f"دیسک {result.get('path')}: {free_gb:.1f} گیگابایت خالی از {total_gb:.1f} گیگابایت."
            ), "tool_disk_result"
        if call.tool == "clipboard_read":
            return (
                f"Clipboard text:\n{result}" if language == "en" else f"متن Clipboard:\n{result}"
            ), "tool_clipboard_result"
        if call.tool == "clipboard_write":
            return (
                "Copied the text to the clipboard." if language == "en" else "متن رو داخل Clipboard گذاشتم."
            ), "tool_clipboard_result"
        if call.tool == "search_files" and isinstance(result, dict):
            items = list(result.get("items", []))
            if not items:
                return (
                    "No file matched all requested filters."
                    if language == "en" else "هیچ فایلی با همهٔ شرط‌های خواسته‌شده پیدا نشد."
                ), "tool_file_search_empty"
            selected = items[0] if isinstance(items[0], dict) else {}
            name = str(selected.get("name", selected.get("path", "")))
            size = int(selected.get("size", 0) or 0)
            return (
                f"Selected {name} ({size:,} bytes) from {result.get('matched', len(items))} match(es)."
                if language == "en"
                else f"از بین {result.get('matched', len(items))} نتیجه، «{name}» با حجم {size:,} بایت انتخاب شد."
            ), "tool_file_search_result"
        if call.tool == "undo_last_action" and isinstance(result, dict):
            if result.get("status") == "nothing_to_undo":
                return (
                    "There is no reversible action to undo."
                    if language == "en" else "عملیات برگشت‌پذیری برای Undo وجود ندارد."
                ), "tool_undo_empty"
            return (
                f"Undid the last action using {result.get('inverse_tool')}."
                if language == "en"
                else f"آخرین عملیات با «{result.get('inverse_tool')}» برگردانده شد."
            ), "tool_undo_result"
        if call.tool == "forget_memory" and isinstance(result, dict):
            removed = int(result.get("removed", 0) or 0)
            return (
                f"Forgot {removed} matching memory record(s)."
                if language == "en" else f"{removed} رکورد حافظهٔ مطابق فراموش شد."
            ), "tool_forget_result"
        if call.tool == "diagnose_system" and isinstance(result, dict):
            checks = list(result.get("checks", []))
            lines = []
            for check in checks:
                if not isinstance(check, dict):
                    continue
                marker = "OK" if check.get("healthy") else "ATTENTION"
                lines.append(f"[{marker}] {check.get('name')}: {check.get('detail')}")
                if check.get("recovery"):
                    lines.append(f"  Recovery: {check.get('recovery')}")
            title = (
                f"Diagnose status: {result.get('status')}"
                if language == "en" else f"وضعیت عیب‌یابی: {result.get('status')}"
            )
            return title + "\n" + "\n".join(lines), "tool_diagnose_result"
        if isinstance(result, FileOperationResult):
            labels = {
                "write": ("Wrote the file.", "فایل رو نوشتم."),
                "append": ("Appended to the file.", "متن رو به فایل اضافه کردم."),
                "copy": ("Copied the file.", "فایل رو کپی کردم."),
                "move": ("Moved the file.", "فایل رو منتقل کردم."),
                "rename": ("Renamed the file.", "نام فایل رو تغییر دادم."),
                "create_file": ("Created the file.", "فایل رو ساختم."),
                "create_folder": ("Created the folder.", "پوشه رو ساختم."),
                "delete_file": ("Deleted the file.", "فایل رو حذف کردم."),
                "delete_folder": ("Deleted the empty folder.", "پوشهٔ خالی رو حذف کردم."),
            }
            pair = labels.get(result.action, ("File operation completed.", "عملیات فایل انجام شد."))
            target = result.destination or result.source
            return f"{pair[0 if language == 'en' else 1]} {target}", "tool_file_operation_result"
        if call.tool == "web_search" and isinstance(result, SearchReport):
            if result.summary:
                confidence = int(round(result.confidence * 100))
                title = (
                    f"Evidence-based result (confidence {confidence}%):"
                    if language == "en"
                    else f"نتیجهٔ مبتنی بر شواهد (اطمینان {confidence}٪):"
                )
                sources_title = "Sources:" if language == "en" else "منابع:"
                maximum_source = max(
                    (passage.source_index for passage in result.passages), default=4
                )
                sources = "\n".join(
                    f"{index}. {item.title} — {item.url}"
                    for index, item in enumerate(result.evidence[:max(4, maximum_source)], 1)
                )
                return f"{title}\n{result.summary}\n\n{sources_title}\n{sources}", "tool_search_result"
            if result.status == "unavailable":
                return (
                    "I could not reach the configured search providers. I will not invent an answer; check the connection or firewall and try again."
                    if language == "en" else
                    "به سرویس‌های جست‌وجوی تنظیم‌شده دسترسی پیدا نکردم. برای جلوگیری از جواب حدسی چیزی نمی‌سازم؛ اتصال اینترنت یا فایروال را بررسی کن و دوباره تلاش کن."
                ), "tool_search_unavailable"
            if result.status == "insufficient":
                return (
                    "Results existed, but their text did not support a reliable answer. Try a more specific query."
                    if language == "en" else
                    "نتیجه‌هایی پیدا شد، اما متن آن‌ها برای یک پاسخ قابل‌اتکا کافی نبود. سؤال را کمی دقیق‌تر بنویس."
                ), "tool_search_insufficient"
            return (
                "No relevant results were found for that query." if language == "en"
                else "برای این عبارت نتیجهٔ مرتبطی پیدا نشد."
            ), "tool_search_result"
        if call.tool == "web_research" and isinstance(result, ResearchReport):
            if not result.summary:
                if result.status == "unavailable":
                    return (
                        "The research providers were unreachable. I will not fill the gap with a guessed answer."
                        if language == "en" else
                        "سرویس‌های تحقیق در دسترس نبودند؛ جای خالی را با پاسخ حدسی پر نمی‌کنم. اتصال را بررسی کن و دوباره امتحان کن."
                    ), "tool_research_unavailable"
                return (
                    "Research did not find enough reliable text evidence."
                    if language == "en"
                    else "تحقیق به شواهد متنیِ قابل‌اتکای کافی نرسید."
                ), "tool_research_result"
            source_lines = "\n".join(
                f"{index}. {source.title} — {source.url}"
                for index, source in enumerate(result.sources[:5], 1)
            )
            if language == "en":
                return (
                    f"Research result (confidence {int(round(result.confidence * 100))}%):\n"
                    f"{result.summary}\n\nSources:\n{source_lines}"
                ), "tool_research_result"
            return (
                f"نتیجهٔ تحقیق (اطمینان {int(round(result.confidence * 100))}٪):\n"
                f"{result.summary}\n\nمنابع:\n{source_lines}"
            ), "tool_research_result"
        if call.tool == "web_search" and isinstance(result, BrowserActionResult):
            query = str(call.arguments.get("query", ""))
            browser_id = str(call.arguments.get("target_browser", ""))
            browser = self.entities.app(browser_id)
            label = browser.name if browser else browser_id or result.label
            if language == "en":
                return f"Searched for “{query}” in {label}.", "tool_browser_search_result"
            return f"«{query}» رو داخل {label} سرچ کردم.", "tool_browser_search_result"
        if call.tool == "ui_open_settings":
            return self._render("tool_success", language, original), "tool_ui_settings"
        if call.tool in {"enumerate_windows", "find_window"} and isinstance(result, tuple):
            if not result:
                return (
                    "No matching visible window was found."
                    if language == "en" else "پنجرهٔ قابل‌مشاهده‌ای پیدا نشد."
                ), "tool_window_list_result"
            lines = [
                f"{index}. {getattr(item, 'title', str(item))}"
                for index, item in enumerate(result[:30], 1)
            ]
            title = "Visible windows:" if language == "en" else "پنجره‌های قابل‌مشاهده:"
            return f"{title}\n" + "\n".join(lines), "tool_window_list_result"
        if call.tool in {"find_installed_app", "list_installed_apps"} and isinstance(result, tuple):
            if not result:
                return (
                    "No matching installed app was found."
                    if language == "en" else "برنامهٔ نصب‌شدهٔ مطابقی پیدا نشد."
                ), "tool_app_index_result"
            lines = [
                f"{index}. {getattr(item, 'name', str(item))}"
                for index, item in enumerate(result[:40], 1)
            ]
            title = "Installed apps:" if language == "en" else "برنامه‌های نصب‌شده:"
            return f"{title}\n" + "\n".join(lines), "tool_app_index_result"
        if call.tool == "run_command" and hasattr(result, "stdout"):
            stdout = str(getattr(result, "stdout", "")).strip()
            stderr = str(getattr(result, "stderr", "")).strip()
            body = stdout or stderr or (
                "Command completed without output."
                if language == "en" else "دستور بدون خروجی تمام شد."
            )
            return body[:24_000], "tool_command_result"
        if call.tool in {
            "browser_open_url", "search_google", "search_youtube", "current_url",
            "navigate_back", "navigate_forward", "refresh_browser", "new_tab",
            "close_tab", "switch_tab", "browser_read_page", "browser_links",
            "browser_open_link", "browser_click", "browser_fill",
        } and hasattr(result, "data"):
            data = getattr(result, "data", {}) or {}
            if call.tool == "browser_read_page" and data.get("text"):
                return str(data["text"]), "tool_browser_page_result"
            if call.tool == "browser_links":
                links = data.get("links", [])
                lines = [
                    f"{index}. {item.get('text') or item.get('url')} — {item.get('url')}"
                    for index, item in enumerate(links[:50], 1)
                ]
                return "\n".join(lines) or (
                    "No links found." if language == "en" else "لینکی پیدا نشد."
                ), "tool_browser_links_result"
            url = str(getattr(result, "url", ""))
            action = str(getattr(result, "action", call.tool)).replace("_", " ")
            if language == "en":
                return styled(f"Browser action completed: {action}.{f' {url}' if url else ''}"), "tool_browser_action_result"
            return styled(f"عمل مرورگر انجام شد: {action}.{f' {url}' if url else ''}"), "tool_browser_action_result"
        if call.tool in {
            "focus_window", "close_window", "minimize_window", "maximize_window",
            "restore_window", "move_window", "resize_window", "minimize_app",
            "maximize_app", "restore_app", "type_text", "press_key", "hotkey",
            "click", "double_click", "right_click", "scroll", "move_mouse",
            "volume_up", "volume_down", "set_volume", "mute", "unmute",
            "set_brightness", "shutdown_system", "restart_system", "sleep_system",
            "logoff_system", "clipboard_clear",
        } and hasattr(result, "action"):
            action = str(getattr(result, "action", call.tool)).replace("_", " ")
            base = (
                f"Completed and checked: {action}."
                if language == "en" else f"انجام شد و بررسی کردم: {action}."
            )
            return styled(base), "tool_action_result"
        if call.tool in {
            "file_info", "folder_info", "battery_info", "network_info", "process_info",
            "cpu_info", "gpu_info", "ram_info",
        }:
            payload = result if isinstance(result, (dict, list, tuple)) else {"result": str(result)}
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)[:24_000], "tool_information_result"
        return self._render("tool_success", language, original), "tool_success"

    @classmethod
    def _resolve_plan_value(cls, value: Any, results: dict[str, Any]) -> Any:
        """Resolve typed references to verified outputs of earlier plan steps."""
        if isinstance(value, dict) and set(value) == {"$ref"}:
            reference = str(value["$ref"]).strip()
            parts = reference.split(".")
            if not parts or parts[0] not in results:
                raise ToolError(f"Plan reference is not available: {reference}")
            current: Any = results[parts[0]]
            for part in parts[1:]:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, (list, tuple)) and part.isdigit():
                    index = int(part)
                    if index >= len(current):
                        raise ToolError(f"Plan reference index is out of range: {reference}")
                    current = current[index]
                else:
                    current = getattr(current, part, None)
                    if current is None:
                        raise ToolError(f"Plan reference cannot be resolved: {reference}")
            return current
        if isinstance(value, dict):
            return {key: cls._resolve_plan_value(item, results) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._resolve_plan_value(item, results) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._resolve_plan_value(item, results) for item in value)
        return value

    def _remember_reversible_action(self, call: ToolCall, result: Any) -> None:
        if not isinstance(result, FileOperationResult) or not result.success:
            return
        destination = str(result.destination)
        source = str(result.source) if result.source is not None else ""
        inverse_tool = ""
        inverse_arguments: dict[str, Any] = {}
        if call.tool in {"create_file", "write_file"}:
            inverse_tool, inverse_arguments = "delete_file", {"path": destination}
        elif call.tool == "create_folder":
            inverse_tool, inverse_arguments = "delete_folder", {"path": destination}
        elif call.tool == "copy_file":
            inverse_tool, inverse_arguments = "delete_file", {"path": destination}
        elif call.tool == "move_file" and source:
            inverse_tool = "move_file"
            inverse_arguments = {"source": destination, "destination": source}
        elif call.tool == "rename_file" and source:
            inverse_tool = "rename_file"
            inverse_arguments = {"source": destination, "new_name": Path(source).name}
        elif call.tool == "append_file" and result.previous_size is not None:
            inverse_tool = "truncate_file"
            inverse_arguments = {"path": destination, "size": result.previous_size}
        if inverse_tool:
            self.memory.record_reversible_action(
                self.session_id, call.tool, call.arguments,
                inverse_tool, inverse_arguments,
            )

    def _execute_plan(
        self, plan: ActionPlan, route: IntentRoute, language: str, confirmed: bool = False,
    ) -> AgentReply:
        self._emit_status("PLANNING")
        calls = [step.tool_call for step in plan.steps if step.tool_call]
        if not calls:
            if plan.needs_input:
                self._pending_input = PendingInput(
                    plan.missing_tool,
                    plan.needs_input,
                    dict(plan.pending_arguments),
                    plan.original_request,
                )
                text = FailureRecovery.message(
                    "missing_argument", language, tool=plan.missing_tool,
                    missing=(plan.needs_input,),
                )
                return self._record_reply(
                    text, "tool_clarification", route, language,
                    data={"plan": plan.to_dict(), "missing": plan.needs_input},
                )
            text = self._render(plan.response_intent or "tool_need_file", language, plan.original_request)
            return self._record_reply(text, plan.response_intent, route, language, data={"plan": plan.to_dict()})
        if plan.dry_run:
            descriptions = [step.description for step in plan.steps]
            prefix = "Dry run — nothing was executed:" if language == "en" else "پیش‌نمایش امن — هیچ کاری اجرا نشد:"
            text = prefix + "\n" + "\n".join(
                f"{index}. {description}" for index, description in enumerate(descriptions, 1)
            )
            return self._record_reply(
                text, "tool_dry_run", route, language,
                data={"plan": plan.to_dict(), "executed": False},
            )
        if any(call.requires_confirmation for call in calls) and not confirmed:
            self._pending_plan = plan
            confirmation_intent = (
                plan.response_intent
                if plan.response_intent in {"tool_confirm_dangerous", "tool_confirm_clear"}
                else "tool_confirm_dangerous"
            )
            text = self._render(confirmation_intent, language, plan.original_request)
            return self._record_reply(
                text, confirmation_intent, route, language,
                data={"plan": plan.to_dict()},
            )
        if any(call.tool in {"web_search", "web_research"} for call in calls) and not self.internet_enabled:
            text = self._render("internet_disabled", language, plan.original_request)
            return self._record_reply(text, "internet_disabled", route, language, data={"plan": plan.to_dict()})

        outputs: list[str] = []
        executed_calls: list[ToolCall] = []
        step_results: dict[str, Any] = {}
        final_intent = "tool_success"
        try:
            for original_call in calls:
                if self._cancel_event.is_set():
                    return self._cancelled_reply(route, language)
                resolved_arguments = self._resolve_plan_value(
                    original_call.arguments, step_results
                )
                if not isinstance(resolved_arguments, dict):
                    raise ToolError("Resolved tool arguments must be an object")
                call = ToolCall(
                    original_call.tool, resolved_arguments,
                    original_call.requires_confirmation, original_call.call_id,
                    original_call.type,
                )
                executed_calls.append(call)
                self._record_decision(plan.original_request, route, call.tool)
                if call.tool == "blocked_dangerous_request":
                    text = (
                        "This build intentionally does not implement destructive system actions."
                        if language == "en"
                        else "این نسخه عمداً ابزار مخرب سیستمی را پیاده‌سازی نکرده است."
                    )
                    outputs.append(text)
                    final_intent = "dangerous_action_blocked"
                    continue
                if call.tool == "ui_clear_conversation":
                    self.new_session()
                    outputs.append(self._render("tool_success", language, plan.original_request))
                    final_intent = "clear_conversation"
                    continue
                status = "WORKING"
                if call.tool in {"web_search", "web_research", "search_google", "search_youtube"}:
                    status = "SEARCHING"
                elif call.tool in {"file_inspect", "browser_read_page", "browser_links"}:
                    status = "READING"
                elif call.tool in {"find_file", "find_folder", "find_installed_app", "find_window"}:
                    status = "FINDING"
                elif call.tool in {"open_app", "open_url", "open_folder", "open_file", "browser_open_url"}:
                    status = "OPENING"
                self._emit_status(status)
                started = time.perf_counter()
                outcome = self.executor.execute(call, confirmed=confirmed)
                latency_ms = (time.perf_counter() - started) * 1000.0
                self.events.tools.write(
                    "tool_result", session_id=self.session_id, tool=call.tool,
                    arguments=call.arguments if self.memory_enabled else {},
                    persistence_disabled=not self.memory_enabled,
                    success=outcome.success, code=outcome.code,
                    confirmed=confirmed, verified=outcome.verification.verified,
                    attempts=outcome.attempts, recovered=outcome.recovered,
                    latency_ms=round(latency_ms, 3),
                )
                self._emit_status("VERIFYING")
                if not outcome.success:
                    if (
                        self.failures is not None
                        and self.memory_enabled
                        and outcome.code != "missing_argument"
                    ):
                        self.failures.add(
                            user_input=plan.original_request,
                            predicted_intent=route.intent,
                            predicted_action=route.action,
                            entity=",".join(route.entities),
                            arguments=call.arguments,
                            tool=call.tool,
                            failure_code=outcome.code,
                            verification=outcome.verification.detail or outcome.verification.code,
                            context=self._task_context(),
                        )
                    if self.debug and outcome.debug_detail:
                        self.logger.debug(
                            "Tool failure detail tool=%s detail=%s", call.tool, outcome.debug_detail
                        )
                    result_label = getattr(outcome.result, "label", "")
                    if not result_label:
                        app_id = str(call.arguments.get("app", ""))
                        app = self.entities.app(app_id)
                        result_label = app.name if app else app_id
                    text = FailureRecovery.message(
                        outcome.code or "tool_failed", language, label=result_label,
                        tool=call.tool, missing=outcome.missing,
                    )
                    return self._record_reply(
                        text,
                        "tool_clarification" if outcome.code == "missing_argument" else "tool_failure",
                        route,
                        language,
                        data={
                            "plan": plan.to_dict(), "failure_code": outcome.code,
                            "verified": outcome.verification.verified,
                        },
                        status="READY" if outcome.code == "missing_argument" else "ERROR",
                    )
                result = outcome.result
                step_results[f"step{len(step_results) + 1}"] = result
                self._remember_reversible_action(call, result)
                self._update_task_context(call, result)
                if self._cancel_event.is_set():
                    return self._cancelled_reply(route, language, tool_finished=True)
                text, final_intent = self._tool_success_text(
                    call, result, language, plan.original_request
                )
                outputs.append(text)
                if call.tool == "ui_open_settings":
                    return self._record_reply(
                        text, final_intent, route, language, event="open_settings",
                        data={"plan": plan.to_dict()},
                    )
            combined_text = self._combine_step_responses(executed_calls, outputs, language)
            return self._record_reply(
                combined_text, final_intent if len(executed_calls) == 1 else "tool_multi_step_result",
                route, language, data={"plan": plan.to_dict(), "step_count": len(executed_calls)},
            )
        except ToolError as exc:
            self.logger.warning("Tool execution failed: %s", exc)
            if self.failures is not None and self.memory_enabled:
                self.failures.add(
                    user_input=plan.original_request,
                    predicted_intent=route.intent,
                    predicted_action=route.action,
                    entity=",".join(route.entities),
                    tool="execution",
                    failure_code="tool_error",
                    verification=str(exc),
                    context=self._task_context(),
                )
            text = FailureRecovery.message("tool_failed", language)
            return self._record_reply(
                text, "tool_error", route, language,
                data={"plan": plan.to_dict()}, status="ERROR",
            )

    def _resolve_pending_input(self, clean: str, language: str) -> AgentReply | None:
        pending = self._pending_input
        if pending is None:
            return None
        normalized = normalize_text(clean)
        if normalized in {"نه", "لغو", "بیخیال", "cancel", "never mind", "no"}:
            self._pending_input = None
            route = IntentRoute("rejection", "conversation", 1.0, 1.0, source="pending_slot")
            return self._record_reply(
                self._render("tool_cancelled", language, clean),
                "tool_cancelled", route, language,
            )
        if ActionParser.has_action(clean):
            self._pending_input = None
            return None

        value = ""
        entity = ""
        if pending.missing == "app":
            match = self.entities.resolve_app(clean)
            if match:
                value, entity = match.entity_id, match.entity_id
            else:
                candidate = re.sub(
                    r"^(?:برنامه|اپلیکیشن|نرم\s*افزار|app|application)\s+", "", normalized
                ).strip(" .،")
                candidate = re.sub(r"\s+(?:رو|را)$", "", candidate).strip()
                if candidate and len(candidate) <= 80:
                    value = entity = candidate
        elif pending.missing in {"path", "folder"}:
            resolved = DriveResolver.resolve(clean) or KnownFolderResolver.resolve(clean)
            if resolved is None:
                context = self._task_context()
                resolved = KnownFolderResolver.named_child(
                    clean, str(context.get("last_folder", ""))
                )
            if resolved:
                value, entity = resolved.path, resolved.entity_id
            else:
                path = Path(clean).expanduser()
                if path.exists() or re.fullmatch(r"[A-Za-z]:[\\/]?", clean.strip()):
                    value = str(path)
                    entity = value
        elif pending.missing == "query":
            value = clean.strip()
        elif pending.missing == "url":
            value = self.entities.explicit_url(clean)

        if not value:
            route = IntentRoute("tool_clarification", "conversation", 0.9, 0.5, source="pending_slot")
            text = FailureRecovery.message(
                "missing_argument", language, tool=pending.tool,
                missing=(pending.missing,),
            )
            return self._record_reply(
                text, "tool_clarification", route, language,
                data={"missing": pending.missing},
            )

        arguments = dict(pending.arguments)
        arguments[pending.missing] = value
        self._pending_input = None
        route = IntentRoute(
            pending.tool, "tool", 0.99, 0.8,
            arguments=arguments, entities=(entity,) if entity else (),
            source="pending_slot_resolution",
        )
        plan = ActionPlan(
            pending.original_request,
            (PlanStep(1, f"Execute resolved tool {pending.tool}", ToolCall(pending.tool, arguments)),),
            f"tool_{pending.tool}_result",
        )
        return self._execute_plan(plan, route, language)

    def _combine_step_responses(
        self, calls: list[ToolCall], outputs: list[str], language: str,
    ) -> str:
        if len(calls) == 2 and calls[0].tool == "open_app" and calls[1].tool == "web_search":
            app_id = str(calls[0].arguments.get("app", ""))
            app = self.entities.app(app_id)
            label = app.name if app else app_id
            if language == "fa" and app_id == "chrome":
                label = "کروم"
            query = str(calls[1].arguments.get("query", ""))
            if language == "en":
                return f"Opened {label} and searched for “{query}”."
            return f"{label} رو باز کردم و «{query}» رو سرچ کردم."
        if len(calls) == 2 and calls[0].tool == "open_app" and calls[1].tool == "open_url":
            return " ".join(value.rstrip() for value in outputs)
        return "\n\n".join(outputs)

    def _pending_response(self, route: IntentRoute, language: str) -> AgentReply | None:
        if self._pending_plan is None:
            return None
        if route.intent == "affirmation":
            plan = self._pending_plan
            self._pending_plan = None
            return self._execute_plan(plan, route, language, confirmed=True)
        if route.intent == "rejection":
            self._pending_plan = None
            text = self._render("tool_cancelled", language, "cancel")
            return self._record_reply(text, "tool_cancelled", route, language)
        self._pending_plan = None
        return None

    def _learn(self, text: str, language: str) -> AgentReply | None:
        instruction = self.learning.parse(text)
        if instruction is None:
            return None
        context = self._task_context()
        action_route = self.router.route(
            instruction.action_text, context, allow_multi=True, use_corrections=False
        )
        project_match = re.search(
            r"(?:پروژه|project)\s+([\w‌-]+).*(?:باز|open)",
            normalize_text(instruction.action_text),
        )
        if project_match and (
            action_route.route_type != "tool"
            or (action_route.intent == "open_folder" and not action_route.arguments.get("path"))
        ):
            action_route = IntentRoute(
                "open_folder", "tool", 0.82, 0.4,
                arguments={"path": project_match.group(1)},
                entities=(project_match.group(1),), source="project_shortcut",
            )
        if action_route.intent == "unknown":
            message = (
                "I understood the trigger, but not the action. Please state the action more explicitly."
                if language == "en"
                else "عبارت محرک را فهمیدم، اما عمل موردنظر روشن نیست؛ لطفاً عمل را دقیق‌تر بگو."
            )
            unknown = IntentRoute("learning_clarification", "conversation", 0.5, 0.1, source="learning_parser")
            return self._record_reply(message, "learning_clarification", unknown, language)
        self.memory.store_correction(
            instruction.trigger, normalize_text(instruction.trigger), action_route.intent,
            action_route.arguments, instruction.source, self.config.learning.max_corrections,
        )
        self._emit_status("LEARNING")
        message = (
            f"Learned: when you say “{instruction.trigger}”, I will use {action_route.intent}."
            if language == "en"
            else f"یاد گرفتم: وقتی بگی «{instruction.trigger}»، عمل {action_route.intent} را انجام می‌دهم."
        )
        learned = IntentRoute(
            "correction_learned", "conversation", 1.0, 1.0,
            arguments=action_route.arguments, source="correction_memory",
        )
        return self._record_reply(
            message, "correction_learned", learned, language, status="LEARNING",
            data={"trigger": instruction.trigger, "route": action_route.to_dict()},
        )

    def respond(self, text: str) -> AgentReply:
        self._cancel_event.clear()
        clean = str(text).strip()
        language = detect_language(clean)
        self._active_constraints = ConstraintExtractor.extract(clean)
        if not clean:
            return AgentReply(
                "Please type a message." if language == "en" else "لطفاً یک پیام بنویس.",
                "empty", 1.0, language,
            )
        with self._lock:
            self._last_language = language
            self._remember_message(
                "user", clean, {"language": language, "personality": self.current_personality}
            )
            learned = self._learn(clean, language)
            if learned:
                return learned

            current_turn = self.memory.turn_count(self.session_id) if self.memory_enabled else 0
            chain_context = self._task_context()
            chain_context["_current_turn"] = current_turn
            # v14: self-contained verifiable reasoning must bypass contextual
            # follow-up inference.  Otherwise words such as «جمله» in a sequence
            # question can be mistaken for a language-example follow-up before
            # the intent router ever sees the question.
            # Bypass contextual follow-up inference only for self-contained
            # transformations. A bare follow-up such as «همان جمله را رسمی‌تر کن»
            # must remain in the conversation chain and operate on prior context.
            explicit_transform_v17 = bool(
                re.search(r"(?:رسمی(?:.?تر)?(?:\s+کن|\s+بنویس)?|روان(?:.?تر)?(?:\s+کن|\s+بنویس)?|بازنویسی|rewrite|rephrase).{0,40}[:：]", clean, re.I)
                or re.search(r"(?:ترجمه\s+کن|به\s+(?:انگلیسی|فارسی)\s+(?:کن|بگو)|translate).{0,80}[:：]", clean, re.I)
            )
            deterministic_reasoning = (
                self._active_constraints.active
                or explicit_transform_v17
                or self.reasoning.local_intelligence.matches(clean)
                or self.reasoning.challenge_reasoner.matches(clean)
            )
            chain = None if deterministic_reasoning else self.conversation_chain.answer(
                clean,
                language,
                chain_context,
                self.memory.facts(100) if self.memory_enabled else {},
            )
            if chain is not None:
                if self.memory_enabled:
                    if chain.fact_key and chain.fact_value:
                        self.memory.set_fact(chain.fact_key, chain.fact_value, 1.0)
                    merged_state = dict(chain_context)
                    merged_state.pop("topic", None)
                    merged_state.pop("_current_turn", None)
                    merged_state.update(chain.state)
                    self.memory.set_context(self.session_id, "conversation", merged_state)
                chain_route = IntentRoute(
                    chain.intent,
                    "conversation",
                    1.0,
                    0.9,
                    source="conversation_chain_v10",
                )
                return self._record_reply(chain.text, chain.intent, chain_route, language)

            pending_reply = self._resolve_pending_input(clean, language)
            if pending_reply:
                return pending_reply

            # v18: a verified semantic frame is stronger evidence than an action-like
            # surface token. Solve it before Dialogue/Action routing so age, work,
            # inventory, ratio, probability and transitive problems cannot be stolen
            # by tool or memory routes. The solver returns None when its slots are not
            # complete, so ordinary desktop commands still continue to the router.
            if self.memory_enabled:
                from jarvis.memory.context_v21 import MemoryContextV21
                remembered=MemoryContextV21(self.memory).recall(clean,language)
                if remembered:
                    memory_route=IntentRoute('reasoned_answer','conversation',.9,.9,source='episodic_memory_v21')
                    return self._record_reply(remembered,'reasoned_answer',memory_route,language,data={'reasoning_source':'episodic_memory_v21','context_is_hypothesis':True})
            early_semantic = self.reasoning.local_intelligence.solve(clean, language)
            if early_semantic is not None and early_semantic.intent in {
                "probability_answer", "word_problem_answer", "reasoned_answer", "coding_answer"
            }:
                semantic_route = IntentRoute(
                    early_semantic.intent, "conversation", early_semantic.confidence, 0.95,
                    source="semantic_ir_v21_pre_dispatch" if 'v21_universal_verified' in early_semantic.checks else "semantic_ir_v20_pre_dispatch" if 'v20_learned_ir' in early_semantic.checks else "semantic_ir_v19_pre_dispatch",
                )
                self._emit_status("THINKING")
                return self._record_reply(
                    early_semantic.text, early_semantic.intent, semantic_route, language,
                    data={
                        "reasoning_source": "semantic_ir_v21" if 'v21_universal_verified' in early_semantic.checks else "semantic_ir_v20" if 'v20_learned_ir' in early_semantic.checks else "semantic_ir_v19",
                        "confidence": early_semantic.confidence,
                        "self_checks": list(early_semantic.checks),
                        "web_bypassed": True,
                        "tool_bypassed": True,
                    },
                )

            task_context = self._task_context()
            task_context["_current_turn"] = current_turn
            contextual_query = self.dialogue_subjects.resolve(clean, task_context)
            effective_text = contextual_query.query
            route = self.router.route(effective_text, task_context)
            route = self._annotate_information_route(route, contextual_query)
            self._remember_information_mode(contextual_query, route)
            self._record_decision(clean, route)
            pending_reply = self._pending_response(route, language)
            if pending_reply:
                return pending_reply
            slots = self._slots(route)

            attachment_text = (
                self._attachment.preview
                if self._attachment is not None and self._attachment.kind == "text"
                else ""
            )
            conversation_evidence = ""
            if not attachment_text and re.search(
                r"(?:اون\s+متن|آن\s+متن|متن\s+بالا|حرف\s+قبلی|پیام\s+قبلی|"
                r"چیزی\s+که\s+گفتم|\b(?:the text above|previous message|what i said|that passage)\b)",
                normalize_text(clean), re.I,
            ):
                for role, content, _metadata in reversed(list(self._short_term)[:-1]):
                    if role == "user" and len(content.strip()) >= 20:
                        conversation_evidence = content
                        break
            grounding_text = attachment_text or conversation_evidence
            supplied_reasoning = self.reasoning.answer_supplied_text(
                clean, language, grounding_text
            )
            if supplied_reasoning is not None:
                self._emit_status("THINKING")
                return self._record_reply(
                    supplied_reasoning.text,
                    supplied_reasoning.intent,
                    route,
                    language,
                    data={
                        "reasoning_source": supplied_reasoning.source,
                        "knowledge_score": supplied_reasoning.knowledge_score,
                        "confidence": supplied_reasoning.confidence,
                        "self_checks": list(supplied_reasoning.checks),
                        "attachment_used": bool(attachment_text),
                        "conversation_context_used": bool(conversation_evidence),
                    },
                )

            context_decision = self.context_engine.analyze(
                clean, language, route.intent, task_context
            )
            if context_decision.handled:
                if self.memory_enabled:
                    merged_state = dict(task_context)
                    merged_state.pop("topic", None)
                    merged_state.pop("_current_turn", None)
                    merged_state.update(context_decision.state)
                    self.memory.set_context(
                        self.session_id, context_decision.topic, merged_state
                    )
                    for key, value in context_decision.facts.items():
                        self.memory.set_fact(key, value, route.confidence)
                slots.update(context_decision.slots)
                if context_decision.response_intent == "context_low_mood":
                    fluent = self.fluency.compose(clean, "user_low_mood", language)
                    if fluent is not None:
                        return self._record_reply(
                            fluent.text, "context_low_mood", route, language,
                            data={
                                "fluency_model": self.fluency.version,
                                "fluency_label": fluent.label,
                                "fluency_confidence": fluent.confidence,
                                "quality_checks": list(fluent.checks),
                            },
                        )
                response_text = self._render(
                    context_decision.response_intent, language, clean, slots
                )
                return self._record_reply(
                    response_text, context_decision.response_intent, route, language
                )

            if route.intent == "multi_step_task":
                plans: list[ActionPlan] = []
                planning_context = dict(task_context)
                planning_context["_planning_chain"] = True
                for segment in route.arguments.get("segments", []):
                    child_route = self.router.route(
                        str(segment), planning_context, allow_multi=False
                    )
                    child_plan = self.planner.plan(
                        str(segment), child_route,
                        self._attachment.path if self._attachment else None,
                    )
                    if child_plan:
                        plans.append(child_plan)
                        if child_plan.needs_input:
                            break
                        for step in child_plan.steps:
                            if step.tool_call:
                                planning_context = TaskContext.update(
                                    planning_context,
                                    step.tool_call.tool,
                                    step.tool_call.arguments,
                                )
                                planning_context["_planning_chain"] = True
                if plans:
                    return self._execute_plan(
                        self.planner.combine(clean, plans), route, language
                    )

            if route.route_type == "tool":
                attachment_path = self._attachment.path if self._attachment else None
                plan = self.planner.plan(clean, route, attachment_path)
                if plan:
                    return self._execute_plan(plan, route, language)

            intent = route.intent
            if intent == "set_personality":
                selected = str(route.arguments.get("personality", "Normal"))
                response_text = self.set_personality(selected)
                return self._record_reply(
                    response_text,
                    "personality_changed",
                    route,
                    language,
                    data={"personality": selected},
                )
            if intent == "set_user_name":
                name = route.slots.get("user_name", "")
                if name and self.memory_enabled:
                    self.memory.set_fact("user_name", name, route.confidence)
                slots["user_name"] = name
            elif intent == "ask_user_name":
                intent = "ask_user_name_known" if slots.get("user_name") else "ask_user_name_unknown"
            elif intent == "greeting" and slots.get("user_name"):
                intent = "greeting_named"

            if intent == "system_assessment":
                self._emit_status("WORKING")
                outcome = self.executor.execute(ToolCall("system_info", {}))
                self._emit_status("VERIFYING")
                if not outcome.success or not isinstance(outcome.result, dict):
                    text = FailureRecovery.message(outcome.code or "tool_failed", language)
                    return self._record_reply(
                        text, "tool_failure", route, language,
                        data={"failure_code": outcome.code}, status="ERROR",
                    )
                self._emit_status("THINKING")
                reasoned = self.reasoning.answer_system_assessment(
                    clean, language, outcome.result
                )
                return self._record_reply(
                    reasoned.text, reasoned.intent, route, language,
                    data={
                        "reasoning_source": reasoned.source,
                        "confidence": reasoned.confidence,
                        "self_checks": list(reasoned.checks),
                        "system_info_used": True,
                    },
                )

            if intent == "writing_request" and self._active_constraints.active:
                self._emit_status("THINKING")
                constrained = ConstraintAwareComposer.compose(clean, language, self._active_constraints)
                return self._record_reply(
                    constrained, "constraint_response", route, language,
                    data={"constraint_composer": "v16", "constraints_active": True},
                )

            if intent == "rewrite_request":
                rewritten = self.rewrite_engine.rewrite(clean, language)
                if rewritten is not None:
                    self._emit_status("THINKING")
                    return self._record_reply(
                        rewritten.text, "rewrite_response", route, language,
                        data={
                            "rewrite_engine": "v18_morphology_fidelity",
                            "fluency_label": "rewrite",
                            "confidence": rewritten.confidence,
                            "quality_checks": list(rewritten.checks),
                        },
                    )

            if intent in {
                "writing_request", "rewrite_request", "advice_request", "short_story", "support_request", "user_low_mood",
            }:
                fluent = self.fluency.compose(clean, intent, language)
                if fluent is not None:
                    self._emit_status("THINKING")
                    return self._record_reply(
                        fluent.text, "fluent_response", route, language,
                        data={
                            "fluency_model": self.fluency.version,
                            "fluency_label": fluent.label,
                            "fluency_confidence": fluent.confidence,
                            "fluency_source": fluent.source,
                            "quality_checks": list(fluent.checks),
                        },
                    )

            if intent == "translation":
                self._emit_status("THINKING")
                translated = self.reasoning.local_intelligence.solve(effective_text, language)
                if translated is not None and translated.intent == "translation_answer":
                    return self._record_reply(
                        translated.text, translated.intent, route, language,
                        data={
                            "reasoning_source": "translation_pipeline_v16",
                            "confidence": translated.confidence,
                            "self_checks": list(translated.checks),
                            "knowledge_rag_bypassed": True,
                        },
                    )
                # Translation is intentionally isolated from Knowledge/RAG. A
                # missing local translation may degrade to a clarification, but
                # it must never retrieve an unrelated knowledge entry.
                fallback = (
                    "برای این جمله ترجمهٔ محلیِ قابل‌اعتماد پیدا نکردم؛ متن را کوتاه‌تر یا با ساختار روشن‌تر بفرست."
                    if language == "fa" else
                    "I could not produce a reliable local translation for that sentence; please rephrase it more directly."
                )
                return self._record_reply(
                    fallback, "translation_clarification", route, language,
                    data={"reasoning_source": "translation_guard_v16", "knowledge_rag_bypassed": True},
                )

            if route.route_type == "think" or intent == "complex_question":
                self._emit_status("THINKING")
                reasoned = self.reasoning.answer_complex(
                    effective_text, language, self.memory.turn_count(self.session_id),
                    task_context, grounding_text,
                )
                if reasoned.confidence < 0.5 and self.neural is not None and route.unknown_reason != "unverifiable_future_claim":
                    history = tuple(content for _, content, _ in list(self._short_term)[-4:])
                    candidate = self.neural.generate(
                        clean, recent_context=history, sampling_profile="Precise",
                        cancel_check=self._cancel_event.is_set,
                    )
                    if candidate.accepted and candidate.confidence >= 0.68:
                        return self._record_reply(
                            candidate.text, "neural_fallback", route, language,
                            data={
                                "reasoning_source": "smart_brain_fallback_v14",
                                "generation_confidence": candidate.confidence,
                                "model": str(self.neural.metadata.get("model_id", "jarvis_smart_brain")),
                            },
                        )
                if reasoned.confidence < 0.5 and self.internet_enabled:
                    research_arguments: dict[str, Any] = {"query": effective_text}
                    if contextual_query.subject:
                        research_arguments["subject"] = contextual_query.subject
                    search_plan = ActionPlan(
                        clean,
                        (
                            PlanStep(1, "Research independent evidence", ToolCall("web_research", research_arguments)),
                        ),
                        "tool_research_result",
                    )
                    return self._execute_plan(search_plan, route, language)
                polished_text = self.fluency.polish_grounded(
                    reasoned.text, clean, language
                )
                return self._record_reply(
                    polished_text, reasoned.intent, route, language,
                    data={
                        "reasoning_source": reasoned.source,
                        "knowledge_score": reasoned.knowledge_score,
                        "confidence": reasoned.confidence,
                        "self_checks": list(reasoned.checks),
                    },
                )

            if intent in {"factual_question", "knowledge_question", "unknown"} or route.route_type in {"information", "unknown"}:
                reasoned = self.reasoning.answer_information(
                    effective_text, language, self.memory.turn_count(self.session_id),
                    self.internet_enabled, grounding_text,
                )
                if (
                    reasoned.source == "confidence_guard"
                    and self.neural is not None
                    and route.unknown_reason != "unverifiable_future_claim"
                    and (route.route_type == "unknown" or route.confidence < 0.58 or route.source in {"fallback", "neural_hybrid"})
                ):
                    history = tuple(content for _, content, _ in list(self._short_term)[-4:])
                    candidate = self.neural.generate(
                        clean, recent_context=history, sampling_profile="Precise",
                        cancel_check=self._cancel_event.is_set,
                    )
                    if candidate.accepted and candidate.confidence >= 0.68:
                        return self._record_reply(
                            candidate.text, "neural_fallback", route, language,
                            data={
                                "reasoning_source": "smart_brain_fallback_v14",
                                "generation_confidence": candidate.confidence,
                                "model": str(self.neural.metadata.get("model_id", "jarvis_smart_brain")),
                            },
                        )
                if (
                    reasoned.source == "confidence_guard"
                    and self.internet_enabled
                    and route.unknown_reason != "unverifiable_future_claim"
                ):
                    search_arguments = {"query": effective_text}
                    if contextual_query.subject:
                        search_arguments["subject"] = contextual_query.subject
                    search_plan = ActionPlan(
                        clean,
                        (
                            PlanStep(1, "Search for grounded information", ToolCall("web_search", search_arguments)),
                        ),
                        "tool_search_result",
                    )
                    return self._execute_plan(search_plan, route, language)
                polished_text = self.fluency.polish_grounded(
                    reasoned.text, clean, language
                )
                return self._record_reply(
                    polished_text, reasoned.intent, route, language,
                    data={
                        "reasoning_source": reasoned.source,
                        "knowledge_score": reasoned.knowledge_score,
                        "confidence": reasoned.confidence,
                        "self_checks": list(reasoned.checks),
                        "conversation_subject_used": contextual_query.used_context,
                        "information_subject": contextual_query.subject,
                    },
                )

            if self.neural is not None and intent == "smalltalk":
                self._emit_status("THINKING")
                history = tuple(
                    content for _, content, _ in list(self._short_term)[-4:]
                )
                candidate = self.neural.generate(
                    clean,
                    recent_context=history,
                    sampling_profile=self.memory.get_setting("sampling_profile", "Balanced") or "Balanced",
                    cancel_check=self._cancel_event.is_set,
                )
                if self._cancel_event.is_set():
                    return self._cancelled_reply(route, language)
                if candidate.accepted:
                    return self._record_reply(
                        candidate.text,
                        "neural_conversation",
                        route,
                        language,
                        data={
                            "model": str(self.neural.metadata.get("model_id", "jarvis_smart_brain")),
                            "generated_tokens": candidate.generated_tokens,
                            "latency_ms": candidate.latency_ms,
                            "tokens_per_second": candidate.tokens_per_second,
                            "generation_confidence": candidate.confidence,
                            "sampling_profile": candidate.sampling_profile,
                        },
                    )
                self.logger.debug("Neural candidate rejected: %s", candidate.reason)
            response_text = self._render(intent, language, clean, slots)
            return self._record_reply(response_text, intent, route, language)

    def new_session(self) -> int:
        with self._lock:
            self.session_id = self.memory.create_new_session()
            self._short_term.clear()
            self._attachment = None
            self._pending_plan = None
            self._pending_input = None
            return self.session_id

    def clear_long_term_memory(self) -> None:
        with self._lock:
            self.memory.clear_long_term_memory()
            self._short_term.clear()
