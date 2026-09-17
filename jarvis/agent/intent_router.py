from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.agent.decision import DecisionEngine
from jarvis.agent.challenge_reasoner import ChallengeReasoner
from jarvis.agent.local_intelligence_v22 import LocalIntelligenceV22
from jarvis.agent.semantic_router_v18 import SemanticIntentRouterV18
from jarvis.agent.cognitive_model import CognitiveSkillModel
from jarvis.brain.model import HybridNeuralBrain, IntentPrediction
from jarvis.entities.resolver import EntityResolver
from jarvis.nlu.action_parser import Action, ActionParser
from jarvis.nlu.entities_v7 import EntityExtractorV7, PersianNumberParser
from jarvis.nlu.reference_resolver import ReferenceResolver
from jarvis.nlu.task_parser import TaskSegmenter
from jarvis.nlu.workflow_parser import FileWorkflowParser
from jarvis.nlu.typo import TypoNormalizer
from jarvis.nlu.semantic_slots_v19 import SemanticSlotBinderV19
from jarvis.tools.calculator import CalculatorTool
from jarvis.tools.drives import DriveResolver, KnownFolderResolver
from jarvis.utils.text import normalize_text

if TYPE_CHECKING:
    from jarvis.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class IntentRoute:
    intent: str
    route_type: str
    confidence: float
    margin: float
    slots: dict[str, str] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)
    entities: tuple[str, ...] = ()
    alternatives: tuple[tuple[str, float], ...] = ()
    source: str = "neural"
    unknown_reason: str = ""
    requires_confirmation: bool = False
    decision_mode: str = "fast"
    action: str = ""
    entity_type: str = ""
    referenced: bool = False
    structured_entities: tuple[dict[str, Any], ...] = ()
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "route_type": self.route_type,
            "confidence": round(self.confidence, 5),
            "margin": round(self.margin, 5),
            "arguments": self.arguments,
            "entities": list(self.entities),
            "source": self.source,
            "requires_confirmation": self.requires_confirmation,
            "decision_mode": self.decision_mode,
            "unknown_reason": self.unknown_reason,
            "action": self.action,
            "entity_type": self.entity_type,
            "referenced": self.referenced,
            "structured_entities": list(self.structured_entities),
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True, slots=True)
class _Signature:
    intent: str
    route_type: str
    priority: int
    boost: float
    patterns: tuple[re.Pattern[str], ...]


class FactExtractor:
    _NAME_TOKEN = r"([A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF][A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06FF'‌-]{1,23})"
    _PATTERNS = (
        re.compile(rf"(?:اسم|نام)\s+من\s+{_NAME_TOKEN}(?:\s+(?:است|هست|ـه))?", re.I),
        re.compile(rf"(?:اسمم|نامم)\s+{_NAME_TOKEN}(?:\s+(?:است|هست|ـه))?", re.I),
        re.compile(rf"من\s+{_NAME_TOKEN}\s+هستم", re.I),
        re.compile(rf"من[وو]\s+{_NAME_TOKEN}\s+صدا\s+کن", re.I),
        re.compile(rf"به\s+من\s+{_NAME_TOKEN}\s+بگو", re.I),
        re.compile(rf"(?:my\s+name\s+is|call\s+me|please\s+call\s+me)\s+{_NAME_TOKEN}", re.I),
        re.compile(rf"remember\s+me\s+as\s+{_NAME_TOKEN}", re.I),
        re.compile(rf"(?:i\s+am|i'm)\s+{_NAME_TOKEN}$", re.I),
    )
    _REJECTED = frozenset(
        {"good", "fine", "okay", "ok", "happy", "sad", "bored", "ready", "here", "tired", "leaving", "چی", "چه", "کی", "چیست", "چیه", "رو", "را"}
    )

    @classmethod
    def user_name(cls, text: str) -> str | None:
        normalized = normalize_text(text)
        for pattern in cls._PATTERNS:
            match = pattern.search(normalized)
            if not match:
                continue
            candidate = match.group(1).strip(" .,!؟?،'\"")
            if candidate.casefold() in cls._REJECTED:
                continue
            if re.fullmatch(cls._NAME_TOKEN, candidate, re.I):
                return candidate.title() if candidate.isascii() else candidate
        return None


class IntentRouter:
    """v0.4 verb-first router with entity and reference resolution before neural fallback."""

    _MULTI = re.compile(r"\s+(?:بعد(?:ش)?|سپس|و\s+بعد|then|and\s+then)\s+", re.I)
    _FOLLOW_UP = re.compile(r"^(?:حالا|خب\s+حالا|بعدش|now|next)\s+", re.I)
    _OPEN_HINT = re.compile(r"(?:باز|بیار|برو|بزن|اجرا|\b(?:open|launch|run|start)\b|take\s+me\s+to)", re.I)
    _ASK_NAME = re.compile(r"(?:اسمم چی|اسم من چی|اسم منو یادت|من\s+کی\s+هستم|what is my name|remember my name|who am i)", re.I)
    _DATE = re.compile(r"(?:ساعت چنده|ساعت سیستم|الان ساعت|چه ساعتی|زمان رو بگو|وقت فعلی|الان چه وقتی|what time|time now|current(?:\s+local)? time|امروز چندم|تاریخ امروز|تاریخ رو بگو|روز امروز|امروز چه روز|what day is it|which day|what date|today's (?:date|weekday)|give me the date|current date|روز هفته)", re.I)
    _SYSTEM = re.compile(r"(?:چقدر رم|cpu من|پردازنده من|چند هسته|64 بیت|حافظه خالی|فضای خالی|مشخصات سیستم|وضعیت منابع|system info|how much ram|my cpu|free disk|64.?bit|hardware)", re.I)
    _FOLDER = re.compile(r"(?:پوشه|فولدر|پروژه|folder|directory|project)", re.I)
    _READ = re.compile(r"(?:بخون|بخوان|ببین|بررسی\s*(?:کن)?|نمایش بده|محتوا|read|inspect|show contents?)", re.I)
    _FIND = re.compile(r"(?:پیدا کن|دنبال.*بگرد|دنبالش بگرد|find|locate|search)", re.I)
    _ZIP = re.compile(r"(?:zip|زیپ|آرشیو|فایل فشرده)", re.I)
    _DANGEROUS = re.compile(r"(?:حذف|پاک\s+کن|\b(?:delete|remove|shutdown|restart|format)\b|ریستارت|خاموش کن|فرمت|dangerous\s+command|command\s+خطرناک)", re.I)
    _CLEAR_CHAT = re.compile(r"(?:گفتگو|مکالمه|چت|conversation|chat).*(?:پاک|clear|reset|new)|(?:پاک|clear|reset|new|start|شروع).*(?:chat|conversation|گفتگو|چت)", re.I)
    _LEARNING = re.compile(r"(?:یاد\s*بگیر|به\s*خاطر\s*بسپار|وقتی\s+(?:میگم|گفتم)|learn\s+that|remember\s+that|when\s+i\s+say)", re.I)
    _IMPOSSIBLE_FUTURE = re.compile(r"(?:قیمت\s+دقیق.*فردا|نتیجه\s+قطعی.*آینده|exact.*tomorrow|tomorrow.*exact|guarantee.*future|predict.*exact|آینده.*تضمین|secret\s+i\s+never\s+told)", re.I)
    _SETTINGS = re.compile(r"^(?:تنظیمات(?:\s+رو)?\s+باز\s+کن|تنظیم\s+شخصیت\s+کجاست|open\s+(?:the\s+)?settings|settings)$", re.I)
    _LOW_MOOD = re.compile(r"(?:حوصله\s+ندارم|روز\s+خوبی\s+نیست|حالم\s+تعریفی\s+نداره|حالم\s+خوب\s+نیست)", re.I)
    _VAGUE_UNKNOWN = re.compile(r"^(?:فلان|یه\s+چیز\s+عجیب|some\s+weird\s+thing)\b", re.I)
    _PROJECT_KIND = re.compile(r"^(?:a\s+small\s+game|یک\s+بازی\s+کوچک|ابزار\s+پایتونی)$", re.I)
    _PRIVACY = re.compile(r"(?:از\s+api\s+استفاده\s+می\s*کنی|cloud\s+api|external\s+api)", re.I)
    _SMALLTALK = re.compile(
        r"^\s*(?:بیکارم|"
        r"حوصله?(?:‌?ا)?م\s+(?:خیلی\s+)?سر\s+رفته(?:\s*[،,]?\s*(?:پیشنهادی\s+داری|چی\s*کار\s+کنم|چه\s+کار\s+کنم))?|"
        r"(?:بیا\s+)?(?:چند\s+دقیقه\s+)?(?:با\s+هم\s+)?(?:می.?خوام\s+(?:کمی\s+)?)?(?:گپ|حرف|صحبت)\s+(?:بزنیم|کنیم)|"
        r"می.?تونی\s+کمکم\s+کنی|چه\s+خبر|یه\s+چیزی\s+بگو|"
        r"فکر\s+می.?کنی(?:\s+امروز)?\s+از\s+کجا\s+شروع\s+کنم|الان\s+می.?تونی\s+کنارم\s+باشی|"
        r"i(?:'m|\s+am)\s+bored(?:\s*[،,]?\s*(?:any\s+ideas?))?|"
        r"i(?:'m|\s+am)\s+feeling\s+(?:a\s+little\s+)?stuck(?:\s+today)?|"
        r"(?:can|could)\s+you\s+help\s+me(?:\s+get\s+started)?|"
        r"(?:let'?s|can\s+we)\s+(?:chat|talk)(?:\s+for\s+(?:a\s+)?bit)?|"
        r"what\s+should\s+we\s+talk\s+about|"
        r"(?:tell\s+me\s+(?:one\s+)?(?:useful\s+)?(?:something|thought)|say\s+something\s+interesting)"
        r")\s*[؟?!.,،]*\s*$",
        re.I,
    )
    _REWRITE_REQUEST = re.compile(
        r"(?:بازنویسی\s+کن|بازنویسیش\s+کن|روان.?تر\s+(?:کن|بنویس)|"
        r"بهترش\s+کن|ویرایش\s+کن|(?:رسمی(?:.?تر)?|صمیمی.?تر|حرفه.?ای.?تر|کوتاه.?تر)\s+(?:کن|بنویس)|"
        r"\b(?:rewrite|rephrase|edit)\b|make\s+(?:it|this).{0,20}(?:formal|clearer|shorter|professional))",
        re.I,
    )
    _WRITING_REQUEST = re.compile(
        r"(?:متن|نوشته|چند\s+خط|پیام|ایمیل|نامه|کپشن|پاراگراف|بیو|توضیحات)"
        r".{0,180}(?:بنویس|می.?نویسی|نگارش|بگو|بساز|درست\s+کن|آماده\s+(?:کن|می.?کنی))|"
        r"(?:بنویس|می.?نویسی|بساز|درست\s+کن|آماده\s+(?:کن|می.?کنی)).{0,180}"
        r"(?:متن|نوشته|چند\s+خط|پیام|ایمیل|نامه|کپشن|پاراگراف|بیو|توضیحات)|"
        r"\b(?:write|draft|create)\b.{0,120}\b(?:message|email|letter|caption|paragraph|text)\b",
        re.I,
    )
    _ADVICE_REQUEST = re.compile(
        r"(?:چه|چی)\s*(?:کار|کاری)\s+کنم|چه\s+پیشنهادی\s+داری|"
        r"از\s+کجا\s+شروع\s+کنم|(?:تمرکز|برنامه.?ریزی|مدیریت\s+زمان).{0,90}"
        r"(?:پیشنهاد|کمک|چطور|چه\s+کار)|"
        r"(?:پیشنهاد|کمک).{0,90}(?:تمرکز|برنامه.?ریزی|مدیریت\s+زمان)",
        re.I,
    )
    _FINGLISH_GREETING = re.compile(r"^(?:sobh\s+bekheir|salam)(?:\s+jarvis)?$", re.I)
    _HOW_ARE_YOU_ALT = re.compile(r"^(?:are\s+you\s+good|you\s+okay)\??$", re.I)
    _LANGUAGE_QUESTION = re.compile(
        r"(?:معنی|معنا|ترجمه|معادل|مخفف).{0,80}(?:انگلیسی|فارسی|کلمه|واژه)|"
        r"(?:انگلیسی|فارسی).{0,80}(?:معنی|معنا|ترجمه|معادل|مخفف)|"
        r"\b(?:translate|translation|meaning|equivalent|abbreviation)\b",
        re.I,
    )
    _COMPLEX_HINT = re.compile(r"(?:reasoned\s+recommendation|analy[sz]e\s+the\s+tradeoffs?)", re.I)
    _PATH = re.compile(r"(?:(?:[A-Za-z]:[\\/]|\.?\.?[\\/]|/)[^\n\r\"']+)")
    _CANONICAL_INTENTS = {
        "ask_time": "date_time",
        "ask_date": "date_time",
        "file_read": "read_file",
        "zip_list": "zip_inspect",
        "internet_search": "web_search",
        "open_browser": "open_default_browser",
    }

    def __init__(
        self,
        brain: HybridNeuralBrain | None,
        signatures_path: Path | None = None,
        entities: EntityResolver | None = None,
        memory: MemoryStore | None = None,
        cognitive_model_path: Path | None = None,
        semantic_router_path: Path | None = None,
    ) -> None:
        self.brain = brain
        self.entities = entities
        self.memory = memory
        self.cognitive = CognitiveSkillModel(
            cognitive_model_path or Path("models/cognitive_skills_v11.json")
        )
        self.semantic = SemanticIntentRouterV18(
            semantic_router_path or Path("models/semantic_router_v18.npz")
        )
        self.slot_binder = SemanticSlotBinderV19()
        self._signatures: tuple[_Signature, ...] = ()
        if signatures_path and signatures_path.is_file():
            with signatures_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            signatures: list[_Signature] = []
            for item in payload.get("signatures", []):
                signatures.append(
                    _Signature(
                        str(item["intent"]), str(item.get("route_type", "conversation")),
                        int(item.get("priority", 0)), float(item.get("boost", 0.7)),
                        tuple(re.compile(str(pattern), re.I) for pattern in item.get("patterns", [])),
                    )
                )
            self._signatures = tuple(sorted(signatures, key=lambda value: value.priority, reverse=True))

    @staticmethod
    def _tool(
        intent: str,
        arguments: dict[str, Any],
        *,
        confidence: float = 0.98,
        source: str = "decision_engine",
        entities: tuple[str, ...] = (),
        confirmation: bool = False,
        mode: str = "fast",
        action: str = "",
        entity_type: str = "",
        referenced: bool = False,
        structured_entities: tuple[dict[str, Any], ...] = (),
        dry_run: bool = False,
    ) -> IntentRoute:
        return IntentRoute(
            intent, "tool", confidence, max(0.1, confidence - 0.55),
            arguments=arguments, entities=entities, source=source,
            requires_confirmation=confirmation, decision_mode=mode,
            action=action, entity_type=entity_type, referenced=referenced,
            structured_entities=structured_entities, dry_run=dry_run,
        )

    @staticmethod
    def _app_query(text: str) -> str:
        normalized = normalize_text(text)
        match = re.search(
            r"(?:برنامه|اپلیکیشن|نرم\s*افزار|app|application)\s+"
            r"([\w.+‌ -]{1,80}?)(?=\s+(?:رو|را|باز|ببند|اجرا|بیار|close|open|launch|start|quit)|$)",
            normalized,
            re.I,
        )
        candidate = match.group(1).strip(" .") if match else ""
        if candidate in {
            "رو", "را", "یه", "یک", "the", "a", "باز", "باز کن", "اجرا", "اجرا کن",
            "open", "launch", "start", "run",
        }:
            return ""
        return candidate

    @staticmethod
    def _bare_app_query(text: str) -> str:
        normalized = normalize_text(text)
        match = re.match(
            r"([a-z][a-z0-9 .+_-]{1,60}?)\s+(?:رو|را)?\s*"
            r"(?:باز|اجرا|بیار|ببند|ریستارت)\b",
            normalized,
            re.I,
        )
        if match:
            return match.group(1).strip(" .")
        prefix = re.match(
            r"(?:open|launch|start|run|close|quit|terminate|restart)\s+"
            r"(?:the\s+)?(?:app(?:lication)?\s+)?([a-z][a-z0-9 .+_-]{1,60})$",
            normalized,
            re.I,
        )
        return prefix.group(1).strip(" .") if prefix else ""

    @staticmethod
    def _context_entity(context: dict[str, Any], key: str) -> tuple[str, str, str]:
        value = context.get(key)
        if isinstance(value, dict):
            return (
                str(value.get("type", "")), str(value.get("id", "")),
                str(value.get("label", "")),
            )
        return "", str(value or ""), ""

    def _search_query(
        self,
        text: str,
        app_alias: str = "",
        website_alias: str = "",
    ) -> str:
        query = normalize_text(text)
        query = re.sub(
            r"(?:سرچ\s*(?:کن|بزن|کنید)?|جستجو\s*(?:کن|کنید)?|گوگل\s*کن|"
            r"بگرد|\bbegard\b|\b(?:search(?:\s+for)?|look\s+up|google)\b)",
            " ", query, flags=re.I,
        )
        for alias in (app_alias, website_alias):
            if alias:
                query = query.replace(alias, " ", 1)
        query = re.sub(
            r"(?:داخلش|توش|درونش|in\s+it|inside\s+it|داخل\s+(?:اون|همون)|"
            r"توی\s+(?:اون|همون))", " ", query, flags=re.I,
        )
        query = query.strip()
        query = re.sub(r"^(?:حالا|خب\s+حالا|بعدش|now|next)\s+", "", query, flags=re.I)
        query = re.sub(r"^(?:تو|توی|داخل|در|روی|on|in|to|about|for)\s+", "", query, flags=re.I)
        query = re.sub(
            r"^(?:وب|اینترنت|the\s+web|web)(?:\s+(?:for|درباره|برای))?\s+",
            "", query, flags=re.I,
        )
        query = re.sub(r"^(?:درباره|راجع\s+به|برای|about|for)\s+", "", query, flags=re.I)
        query = re.sub(r"\s+(?:رو|را)\s*$", "", query)
        return re.sub(r"\s+", " ", query).strip(" :،؟?")

    def _semantic_action_route(
        self,
        text: str,
        normalized: str,
        context: dict[str, Any],
    ) -> IntentRoute | None:
        if not self.entities:
            return None
        action_match = ActionParser.parse(text)
        action = action_match.action
        if action is Action.UNKNOWN:
            return None

        app = self.entities.resolve_app(text)
        website = self.entities.resolve_website(text)
        explicit_url = self.entities.explicit_url(text)
        drive = DriveResolver.resolve(text)
        folder = KnownFolderResolver.resolve(text)
        if folder is None:
            base = str(context.get("last_folder", ""))
            folder = KnownFolderResolver.named_child(text, base)
        referenced = ReferenceResolver.resolve(
            text, context, ("app", "website", "url", "folder", "drive", "file")
        )
        generic_browser = bool(re.search(r"(?:مرورگر|browser)", normalized, re.I))
        explicit_website = bool(re.search(r"(?:سایت|وب\s*سایت|website|web\s+site)", normalized, re.I))
        generic_folder = bool(re.search(r"(?:یه|یک|a|the)?\s*(?:پوشه|فولدر|folder|directory)", normalized, re.I))
        generic_app = bool(re.search(r"(?:برنامه|اپلیکیشن|نرم\s*افزار|\bapp(?:lication)?\b)", normalized, re.I))

        if action in {Action.MINIMIZE, Action.MAXIMIZE, Action.RESTORE}:
            intent_by_action = {
                Action.MINIMIZE: "minimize_app",
                Action.MAXIMIZE: "maximize_app",
                Action.RESTORE: "restore_app",
            }
            app_id = app.entity_id if app else ""
            used_reference = False
            if not app_id and referenced and referenced.entity_type == "app":
                app_id, used_reference = referenced.entity_id, True
            if not app_id:
                for key in ("last_active_app", "last_opened_app", "last_browser"):
                    _kind, candidate, _label = self._context_entity(context, key)
                    if candidate:
                        app_id, used_reference = candidate, True
                        break
            if app_id:
                return self._tool(
                    intent_by_action[action], {"app": app_id}, source="window_context_v6",
                    entities=(app_id,), action=action.value, entity_type="app",
                    referenced=used_reference,
                )
            _kind, _handle, window_label = self._context_entity(context, "last_window")
            window_query = window_label or str(context.get("last_window_title", ""))
            return self._tool(
                intent_by_action[action].replace("_app", "_window"),
                {"window": window_query}, source="window_context_v6",
                action=action.value, entity_type="window", referenced=bool(window_query),
            )

        if action is Action.CLOSE and re.search(
            r"(?:همه|همشون|هر\s*چی|تمام|all|every).*(?:مرورگر|browser)",
            normalized,
            re.I,
        ):
            return self._tool(
                "close_all_browsers", {}, source="browser_collection_resolver",
                action=action.value, entity_type="browser_collection",
            )

        if action is Action.CLOSE and generic_browser and re.search(
            r"(?:پیش\s*فرض|پیشفرض|default)", normalized, re.I,
        ):
            return self._tool(
                "close_default_browser", {}, source="default_browser_resolver",
                action=action.value, entity_type="browser",
            )

        if action is Action.DELETE:
            if re.search(
                r"(?:همه|تمام|کل|هر\s*چی|all|every|entire|\*).*(?:فایل|پوشه|folder|file)"
                r"|(?:فایل|پوشه|folder|file).*(?:همه|تمام|کل|all|every|entire|\*)",
                normalized,
                re.I,
            ):
                return IntentRoute(
                    "dangerous_request", "tool", 0.99, 0.9,
                    arguments={"request": text.strip()}, source="bulk_delete_safety_gate_v6",
                    requires_confirmation=True, action=action.value,
                )
            path_match = self._PATH.search(text)
            explicit_path = path_match.group(0).strip() if path_match else ""
            if re.search(r"(?:پوشه|فولدر|folder|directory)", normalized, re.I):
                path = explicit_path or str(context.get("last_folder", ""))
                return self._tool(
                    "delete_folder", {"path": path}, source="destructive_file_router_v6",
                    confirmation=True, action=action.value, entity_type="folder",
                    referenced=not bool(explicit_path) and bool(path),
                )
            if re.search(r"(?:فایل|file|\.\w{1,12}\b)", normalized, re.I):
                last_file = context.get("last_file", "")
                if isinstance(last_file, dict):
                    last_file = last_file.get("id", "")
                path = explicit_path or str(last_file or "")
                return self._tool(
                    "delete_file", {"path": path}, source="destructive_file_router_v6",
                    confirmation=True, action=action.value, entity_type="file",
                    referenced=not bool(explicit_path) and bool(path),
                )
            return IntentRoute(
                "dangerous_request", "tool", 0.99, 0.9,
                arguments={"request": text.strip()}, source="action_polarity_guard",
                requires_confirmation=True, action=action.value,
            )

        if action is Action.CREATE:
            request = EntityExtractorV7.file_request(text, context)
            if request is not None:
                structured = tuple(
                    entity.to_dict() for entity in EntityExtractorV7.extract(text, context)
                )
                dry_run = EntityExtractorV7.is_dry_run(text)
                if request.entity_type == "folder":
                    return self._tool(
                        "create_folder", {"path": request.path},
                        source="structured_file_nlu_v7", action=action.value,
                        entity_type="folder", entities=(request.name,) if request.name else (),
                        referenced=request.referenced, structured_entities=structured,
                        dry_run=dry_run,
                    )
                return self._tool(
                    "create_file", {"path": request.path, "content": request.content},
                    source="structured_file_nlu_v7", action=action.value,
                    entity_type="file", entities=(request.name,) if request.name else (),
                    referenced=request.referenced, structured_entities=structured,
                    dry_run=dry_run,
                )

        if action in {Action.WRITE, Action.APPEND} and re.search(r"(?:فایل|file|\.\w{1,12}\b)", normalized, re.I):
            path_match = self._PATH.search(text)
            explicit_path = path_match.group(0).strip() if path_match else ""
            last_file = context.get("last_file", "")
            if isinstance(last_file, dict):
                last_file = last_file.get("id", "")
            path = explicit_path or str(last_file or "")
            content_match = re.search(
                r"(?:بنویس|اضافه\s*(?:کن)?|write|append)(?:\s+(?:داخل|به|to|in))?.*?(?:[:=]|که)\s*(.+)$",
                text, re.I,
            )
            content = content_match.group(1).strip() if content_match else ""
            intent = "append_file" if action is Action.APPEND else "write_file"
            return self._tool(
                intent, {"path": path, "content": content}, source="file_write_router_v6",
                confirmation=True, action=action.value, entity_type="file",
            )

        if action in {Action.COPY, Action.MOVE, Action.RENAME} and re.search(r"(?:فایل|file|\.\w{1,12}\b)", normalized, re.I):
            last_file = context.get("last_file", "")
            if isinstance(last_file, dict):
                last_file = last_file.get("id", "")
            quoted = [value.strip() for value in re.findall(r"[\"«']([^\"»']+)[\"»']", text)]
            path_match = self._PATH.search(text)
            source_path = (quoted[0] if quoted else path_match.group(0).strip() if path_match else str(last_file or ""))
            if action is Action.RENAME:
                name_match = re.search(
                    r"(?:به|to|اسم(?:ش)?\s*(?:رو)?\s*(?:بذار|بگذار))\s+[\"«']?([\w .-]+\.[A-Za-z0-9]{1,12})",
                    text, re.I,
                )
                new_name = quoted[1] if len(quoted) > 1 else name_match.group(1).strip() if name_match else ""
                return self._tool(
                    "rename_file", {"source": source_path, "new_name": new_name},
                    source="file_mutation_router_v6", confirmation=True,
                    action=action.value, entity_type="file",
                )
            destination = quoted[1] if len(quoted) > 1 else ""
            return self._tool(
                "copy_file" if action is Action.COPY else "move_file",
                {"source": source_path, "destination": destination},
                source="file_mutation_router_v6", confirmation=True,
                action=action.value, entity_type="file",
            )

        if action is Action.RUN:
            command_match = re.search(
                r"(?:دستور|command)(?:\s+(?:زیر|این))?\s*[:=]?\s*[\"'«]?(.+?)[\"'»]?$",
                text, re.I,
            )
            command = command_match.group(1).strip() if command_match else ""
            command = re.sub(
                r"\s+(?:(?:رو|را)\s*)?(?:اجرا\s*(?:کن|کنید)?|بزن)\s*$",
                "", command, flags=re.I,
            ).strip()
            command = re.sub(r"\s+(?:please|now)\s*$", "", command, flags=re.I).strip()
            return self._tool(
                "run_command", {"command": command}, source="terminal_router_v6",
                action=action.value, entity_type="command",
            )

        if action is Action.RESTART and re.search(
            r"(?:سیستم|کامپیوتر|رایانه|ویندوز|دستگاه|computer|system|windows|pc)",
            normalized,
            re.I,
        ):
            return IntentRoute(
                "dangerous_request", "tool", 0.99, 0.9,
                arguments={"request": text.strip()}, source="action_polarity_guard",
                requires_confirmation=True, action=action.value, entity_type="system",
            )

        if action in {Action.CLOSE, Action.FOCUS, Action.RESTART, Action.IS_RUNNING}:
            app_id = app.entity_id if app else ""
            label = app.name if app else ""
            used_reference = False
            if not app_id and referenced and referenced.entity_type == "app":
                app_id, label, used_reference = referenced.entity_id, referenced.label, True
            if not app_id and generic_browser:
                _, app_id, label = self._context_entity(context, "last_browser")
                used_reference = bool(app_id)
            if not app_id and generic_app:
                app_id = self._app_query(text)
            if not app_id:
                app_id = self._bare_app_query(text)
            intent_by_action = {
                Action.CLOSE: "close_app",
                Action.FOCUS: "focus_app",
                Action.RESTART: "restart_app",
                Action.IS_RUNNING: "is_app_running",
            }
            intent = intent_by_action[action]
            return self._tool(
                intent, {"app": app_id}, source="verb_first_nlu",
                entities=(app_id,) if app_id else (), action=action.value,
                entity_type="app", referenced=used_reference,
            )

        if action is Action.OPEN:
            named_folder = re.search(
                r"(?:پوشه|فولدر|folder)\s+(.{1,100}?)(?=\s+(?:رو|را)?\s*(?:داخل|توی|تو|در|in)\s+(?:درایو|drive|[A-Za-z]:?)|\s+(?:رو|را)?\s*(?:باز|open)|$)",
                normalized,
                re.I,
            )
            if named_folder and drive:
                name = named_folder.group(1).strip(" .،")
                if name and name.casefold() not in {"یه", "یک", "a", "the"}:
                    return self._tool(
                        "find_and_open_folder", {"root": drive.path, "name": name},
                        source="folder_intelligence_v5", entities=(drive.entity_id, name),
                        action=action.value, entity_type="folder",
                    )
            if drive:
                return self._tool(
                    "open_folder", {"path": drive.path}, source="drive_resolver",
                    entities=(drive.entity_id,), action=action.value, entity_type=drive.entity_type,
                )
            if app and not (website and explicit_website):
                return self._tool(
                    "open_app", {"app": app.entity_id}, source="verb_first_nlu",
                    entities=(app.entity_id,), confidence=app.score,
                    action=action.value, entity_type="app",
                )
            if folder:
                return self._tool(
                    "open_folder", {"path": folder.path}, source="known_folder_resolver",
                    entities=(folder.entity_id,), action=action.value, entity_type="folder",
                )
            if explicit_url:
                browser = ""
                if referenced and referenced.entity_type == "app":
                    browser = referenced.entity_id
                args = {"url": explicit_url}
                if browser:
                    args["browser"] = browser
                return self._tool(
                    "open_url", args, source="url_extractor", action=action.value,
                    entity_type="url", referenced=bool(browser),
                )
            if website:
                assert website.website is not None
                browser = ""
                if referenced and referenced.entity_type == "app":
                    browser = referenced.entity_id
                elif context.get("_planning_chain"):
                    entity_type, entity_id, _ = self._context_entity(context, "last_entity")
                    if entity_type == "app" and entity_id in {"chrome", "edge", "firefox"}:
                        browser = entity_id
                args = {"url": website.website.url, "website": website.entity_id}
                if browser:
                    args["browser"] = browser
                return self._tool(
                    "open_url", args, source="verb_first_nlu",
                    entities=(website.entity_id,), confidence=website.score,
                    action=action.value, entity_type="website", referenced=bool(browser),
                )
            if referenced:
                if referenced.entity_type == "app":
                    return self._tool(
                        "open_app", {"app": referenced.entity_id}, source="reference_resolver",
                        entities=(referenced.entity_id,), action=action.value,
                        entity_type="app", referenced=True,
                    )
                if referenced.entity_type in {"folder", "drive"}:
                    path = referenced.label or referenced.entity_id
                    return self._tool(
                        "open_folder", {"path": path}, source="reference_resolver",
                        entities=(referenced.entity_id,), action=action.value,
                        entity_type=referenced.entity_type, referenced=True,
                    )
                if referenced.entity_type in {"url", "website"}:
                    return self._tool(
                        "open_url", {"url": referenced.entity_id}, source="reference_resolver",
                        action=action.value, entity_type=referenced.entity_type, referenced=True,
                    )
            if generic_browser:
                return self._tool(
                    "open_default_browser", {}, source="verb_first_nlu",
                    action=action.value, entity_type="browser",
                )
            if generic_folder:
                return self._tool(
                    "open_folder", {"path": ""}, source="argument_needed",
                    action=action.value, entity_type="folder",
                )
            if generic_app:
                app_query = self._app_query(text)
                return self._tool(
                    "open_app", {"app": app_query}, source="dynamic_app_query",
                    entities=(app_query,) if app_query else (), action=action.value,
                    entity_type="app",
                )
            bare_app = self._bare_app_query(text)
            if bare_app:
                return self._tool(
                    "open_app", {"app": bare_app}, source="dynamic_app_query",
                    entities=(bare_app,), action=action.value, entity_type="app",
                )

        if action is Action.FIND and re.search(r"(?:پوشه|فولدر|folder|directory)", normalized, re.I):
            root = drive.path if drive else str(
                context.get("last_drive") or context.get("last_folder") or ""
            )
            named = re.search(
                r"(?:پوشه|فولدر|folder|directory)\s+(.{1,100}?)(?=\s+(?:رو|را)?\s*(?:داخل|توی|تو|در|in|پیدا|find)|$)",
                normalized,
                re.I,
            )
            name = named.group(1).strip(" .،") if named else ""
            return self._tool(
                "find_folder", {"root": root, "name": name},
                source="folder_intelligence_v5", action=action.value,
                entity_type="folder", entities=(name,) if name else (),
            )

        if action is Action.FIND and re.search(r"(?:فایل|file)", normalized, re.I):
            folder_path = str(context.get("last_folder", ""))
            file_match = re.search(r"[\w.-]+\.[A-Za-z0-9]{1,12}", normalized)
            named = re.search(
                r"(?:فایل|file)\s+([\w.*-]{1,60})(?=\s+(?:رو|را|پیدا|find|locate)|$)",
                normalized,
                re.I,
            )
            query = file_match.group(0) if file_match else named.group(1) if named else "file"
            return self._tool(
                "find_file", {"folder": folder_path, "query": query},
                source="file_action_router", action=action.value, entity_type="file",
            )

        if action is Action.SEARCH:
            if re.search(r"(?:فایل|پوشه|فولدر|پروژه|file|folder|directory|project)", normalized, re.I):
                return None
            target_browser = ""
            target_website = website
            used_reference = False
            if app and app.entity_id in {"chrome", "edge", "firefox"}:
                target_browser = app.entity_id
            elif referenced and referenced.entity_type == "app":
                target_browser, used_reference = referenced.entity_id, True
            elif referenced and referenced.entity_type == "website" and not target_website:
                target_website = self.entities.resolve_website(referenced.entity_id)
                used_reference = True
            elif context.get("_planning_chain"):
                entity_type, entity_id, _ = self._context_entity(context, "last_entity")
                if entity_type == "app" and entity_id in {"chrome", "edge", "firefox"}:
                    target_browser = entity_id
                elif entity_type == "website":
                    target_website = self.entities.resolve_website(entity_id)
            query = self._search_query(
                text, app.alias if app else "", target_website.alias if target_website else ""
            )
            if target_website and target_website.entity_id == "youtube":
                assert target_website.website is not None
                args: dict[str, Any] = {
                    "query": query,
                    "url": target_website.website.search_address(query),
                    "website": "youtube",
                }
                if target_browser:
                    args["target_browser"] = target_browser
                return self._tool(
                    "youtube_search", args, source="semantic_search",
                    entities=("youtube",), action=action.value, entity_type="website",
                    referenced=used_reference, mode="search",
                )
            args = {"query": query}
            if target_browser:
                args["target_browser"] = target_browser
            return self._tool(
                "web_search", args, source="semantic_search",
                entities=(target_browser,) if target_browser else (),
                action=action.value, entity_type="browser" if target_browser else "information",
                referenced=used_reference, mode="search",
            )
        return None

    def _signature_match(self, normalized: str) -> _Signature | None:
        for signature in self._signatures:
            if any(pattern.search(normalized) for pattern in signature.patterns):
                return signature
        return None

    def _from_correction(self, text: str) -> IntentRoute | None:
        if not self.memory:
            return None
        rule = self.memory.find_correction(text)
        if not rule:
            return None
        tool_intents = {
            "open_url", "open_app", "open_default_browser", "web_search",
            "youtube_search", "read_file", "find_file", "list_folder",
            "zip_inspect", "system_info", "calculator", "date_time", "open_folder",
            "close_app", "close_default_browser", "minimize_app", "maximize_app", "restore_app",
            "create_folder", "create_file", "write_file", "append_file",
            "delete_file", "delete_folder", "run_command", "web_research",
            "shutdown_system", "restart_system", "sleep_system", "logoff_system",
            "set_volume", "volume_up", "volume_down", "mute", "unmute",
        }
        return IntentRoute(
            rule.intent, "tool" if rule.intent in tool_intents else "conversation",
            1.0, 1.0, arguments=dict(rule.arguments), source="correction_memory",
            entities=tuple(str(value) for key, value in rule.arguments.items() if key in {"app", "website"}),
        )

    def _entity_route(self, text: str, normalized: str, context: dict[str, Any]) -> IntentRoute | None:
        if not self.entities:
            return None
        explicit_url = self.entities.explicit_url(text)
        if explicit_url and self._OPEN_HINT.search(normalized):
            return self._tool("open_url", {"url": explicit_url}, source="url_extractor")

        if self.entities.is_default_browser_request(text):
            return self._tool("open_default_browser", {}, source="default_browser_resolver")

        opened = self.entities.resolve_open(text)
        website_alias_only = normalized in {
            alias for website in self.entities.websites for alias in website.aliases
        }
        follow_up = bool(self._FOLLOW_UP.search(normalized)) or website_alias_only
        if not opened and (follow_up or str(context.get("last_action", "")).startswith("open")):
            app = self.entities.resolve_app(text)
            website = self.entities.resolve_website(text)
            opened = app if app and app.entity_id == "chrome" else website or app
        if opened:
            if opened.kind == "app":
                return self._tool(
                    "open_app", {"app": opened.entity_id}, source="app_resolver",
                    entities=(opened.entity_id,), confidence=opened.score,
                )
            assert opened.website is not None
            return self._tool(
                "open_url", {"url": opened.website.url, "website": opened.entity_id},
                source="website_resolver", entities=(opened.entity_id,), confidence=opened.score,
            )

        query, search_site = self.entities.extract_search(text)
        if query:
            if search_site and search_site.entity_id == "youtube":
                assert search_site.website is not None
                return self._tool(
                    "youtube_search", {"query": query, "url": search_site.website.search_address(query)},
                    source="entity_search", entities=("youtube",),
                )
            if search_site and search_site.entity_id not in {"google"}:
                assert search_site.website is not None
                return self._tool(
                    "open_url", {"url": search_site.website.search_address(query), "query": query},
                    source="entity_search", entities=(search_site.entity_id,),
                )
            return self._tool("web_search", {"query": query}, source="search_extractor", mode="search")
        return None

    def route(
        self,
        text: str,
        context: dict[str, Any] | None = None,
        *,
        allow_multi: bool = True,
        use_corrections: bool = True,
    ) -> IntentRoute:
        text = TypoNormalizer.correct(text)
        normalized = normalize_text(text)
        # v19.1: semantic slot binding no longer owns routing. High-confidence
        # system actions and dedicated probability/sequence paths keep precedence;
        # model-backed IR is consumed by LocalIntelligenceV21 in core/reasoning.
        current = context or {}
        if use_corrections:
            corrected = self._from_correction(normalized)
            if corrected:
                return corrected

        # v18: explicit arithmetic is deterministic and must bypass learned
        # semantic routing/brain inference. Keep programming assignments and
        # code snippets out of this fast path so Code Trace retains precedence.
        if CalculatorTool.looks_like_calculation(text) and not re.search(
            r"```|[;{}]|\b(?:def|for|while|print|return|import|class|range)\b|\b[A-Za-z_]\w*\s*=",
            text, re.I,
        ):
            return self._tool(
                "calculator", {"expression": CalculatorTool.extract_expression(text)},
                source="calculator_pre_semantic_v18",
            )

        # High-precision reasoning puzzles must outrank words that look like
        # desktop actions (e.g. «اتاق بسته» != close-app) and generic math.
        if ChallengeReasoner.matches(text):
            skill, model_confidence, model_margin = self.cognitive.predict(text)
            model_verified = (
                self.cognitive.ready
                and skill not in {"", "general"}
                and model_confidence >= 0.12
                and model_margin >= 0.035
            )
            return IntentRoute(
                "complex_question", "think", 0.99, max(0.82, model_margin),
                arguments={"question": text.strip(), "cognitive_skill": skill if model_verified else ""},
                source="trained_cognitive_router_v11" if model_verified else "local_challenge_guard_v15",
                decision_mode="think",
            )

        # v16: a single writing request with output constraints is not a
        # multi-step workflow merely because it contains "and/و".  Route it as
        # one constrained generation task before TaskSegmenter can split it.
        constraint_writing_v16 = bool(
            re.search(r"(?:بنویس|بازنویسی|توضیح\s+بده|write|rewrite|describe|explain)", normalized, re.I)
            and re.search(
                r"(?:دقیقاً|دقیقا|حداکثر|فقط\s+(?:فارسی|انگلیسی)|حتماً|حتما|شامل|حاوی|"
                r"exactly\s+\w+\s+sentences?|at\s+most\s+\d+\s+words?|only\s+(?:persian|english)|"
                r"must\s+(?:include|contain|have)|make\s+sure.{0,30}(?:include|contain|have))",
                normalized, re.I,
            )
        )
        if constraint_writing_v16:
            return IntentRoute(
                "writing_request", "conversation", 0.995, 0.86,
                arguments={"request": text.strip()}, source="constraint_writing_guard_v16",
                decision_mode="fast",
            )

        if allow_multi:
            early_segments = TaskSegmenter.split(text)
            if len(early_segments) >= 2:
                return IntentRoute(
                    "multi_step_task", "tool", 0.96, 0.55,
                    arguments={"segments": list(early_segments)}, source="task_segmenter_v4", decision_mode="think",
                )

        if EntityExtractorV7.is_educational(text) and re.search(
            r"(?:پاک\s+کردن|حذف\s+کردن|delete|remove).{0,50}(?:فایل|file|پوشه|folder|directory)",
            normalized, re.I,
        ):
            return IntentRoute(
                "knowledge_question", "information", 0.99, 0.80,
                arguments={"question": text.strip()}, source="educational_polarity_guard_v15",
                structured_entities=tuple(
                    entity.to_dict() for entity in EntityExtractorV7.extract(text, current)
                ),
            )

        # v18 safety precedence: an explicit dangerous-command request must
        # never be reinterpreted by a learned cognitive label.
        if re.search(r"(?:dangerous\s+command|command\s+خطرناک|دستور\s+خطرناک)", normalized, re.I):
            return IntentRoute(
                "dangerous_request", "tool", 0.99, 0.9,
                arguments={"request": text.strip()}, source="safety_pre_semantic_v18",
                requires_confirmation=True,
            )

        # v15 semantic cognition routing with action-safe precedence.
        # Deterministic semantic guards (translation/rewrite/code/reasoning cues)
        # may outrank action words that occur inside the payload.  A prediction
        # coming only from the learned classifier does NOT outrank an explicit,
        # already-resolved app/file command; this keeps short commands such as
        # «وی اس کد رو بیار» and “launch Calculator” reliable.
        semantic = self.semantic.predict(text)
        semantic_map = {
            "rewrite": ("rewrite_request", "conversation"),
            "constraint_writing": ("writing_request", "conversation"),
            "coding": ("coding", "think"),
            "code_trace": ("code_trace", "think"),
            "translation": ("translation", "think"),
            "probability": ("probability", "think"),
            "logic": ("logic", "think"),
            "word_problem": ("word_problem", "think"),
            "math": ("math", "think"),
        }

        def _semantic_route(prediction: Any) -> IntentRoute | None:
            if prediction is None or prediction.intent in {"desktop_command", "fresh_information", "general_question"}:
                return None
            mapped = semantic_map.get(prediction.intent)
            if mapped is None:
                return None
            intent, route_type = mapped
            return IntentRoute(
                intent, route_type, prediction.confidence, prediction.margin,
                arguments={"question": text.strip(), "semantic_intent": prediction.intent},
                alternatives=prediction.alternatives, source=prediction.source,
                decision_mode="think" if route_type == "think" else "fast",
            )

        early_action = self._semantic_action_route(text, normalized, current)
        guarded_route = (
            _semantic_route(semantic)
            if semantic is not None and semantic.source in {"semantic_guard_v15", "semantic_guard_v17"}
            else None
        )
        # «وی اس کد» contains the standalone Persian token «کد», which is a
        # legitimate programming cue but also part of an app name.  When the
        # mature action parser has already resolved an app command and there is
        # no authoring/programming verb, prefer the app action.  Strong rewrite,
        # translation, code-trace and real coding guards still outrank action
        # words embedded inside their payload.
        app_name_collision = bool(
            guarded_route is not None
            and semantic is not None
            and semantic.intent == "coding"
            and early_action is not None
            and early_action.entity_type == "app"
            and not re.search(
                r"(?:بنویس|بساز|پیاده.?سازی|تابع|اسکریپت|الگوریتم|\bwrite\b|\bcreate\b|\bimplement\b|\bfunction\b|\bscript\b|\bpython\s+code\b)",
                normalized, re.I,
            )
        )
        if guarded_route is not None and not app_name_collision:
            return guarded_route

        if early_action is not None and early_action.source != "dynamic_app_query" and early_action.intent in {
            "open_app", "open_url", "open_folder", "close_app", "focus_app",
            "restart_app", "is_app_running", "delete_file", "delete_folder",
            "create_file", "create_folder", "write_file", "append_file",
            "copy_file", "move_file", "rename_file", "minimize_app",
            "maximize_app", "restore_app", "close_all_browsers",
            "close_default_browser",
        }:
            return early_action

        learned_semantic_route = _semantic_route(semantic)
        if learned_semantic_route is not None and semantic is not None:
            # A learned-only label may assist dispatch, but may not invent a
            # cognitive task without surface/structural evidence. Deterministic
            # semantic guards above remain authoritative. This prevents generic
            # comparisons, error explanations and safety phrases from being
            # stolen by code/math classes merely because of vocabulary overlap.
            learned_evidence = {
                "coding": bool(self.semantic._CODING_CUE.search(normalized) and re.search(r"(?:بنویس|بساز|ایجاد|پیاده.?سازی|write|create|implement|function|تابع)", normalized, re.I)),
                "code_trace": bool(self.semantic._CODE_SYNTAX.search(text) and self.semantic._TRACE_CUE.search(normalized)),
                "translation": bool(self.semantic._TRANSLATION_CUE.search(normalized)),
                "rewrite": bool(self.semantic._REWRITE_CUE.search(normalized)),
                "probability": bool(self.semantic._PROB_CUE.search(normalized)),
                "logic": bool(self.semantic._LOGIC_CUE.search(normalized)),
                "word_problem": bool(self.semantic._WORD_PROBLEM_CUE.search(normalized)),
                "math": bool(self.semantic._MATH_CUE.search(normalized)),
                "constraint_writing": bool(self.semantic._CONSTRAINT_CUE.search(normalized)),
            }.get(semantic.intent, False)
            if learned_evidence:
                return learned_semantic_route

        # v15 high-precision local intelligence runs before generic language/web routing.
        if LocalIntelligenceV22.matches(text):
            return IntentRoute(
                "local_intelligence", "think", 0.995, 0.88,
                arguments={"question": text.strip()}, source="local_intelligence_router_v19",
                decision_mode="think",
            )

        # Reasoning puzzles and quantitative questions must not collide with
        # date/time, close-app or web-search verbs such as «بسته» and «پیدا کن».
        if ChallengeReasoner.matches(text):
            skill, model_confidence, model_margin = self.cognitive.predict(text)
            model_verified = (
                self.cognitive.ready
                and skill not in {"", "general"}
                and model_confidence >= 0.12
                and model_margin >= 0.035
            )
            return IntentRoute(
                "complex_question", "think", 0.99, max(0.82, model_margin),
                arguments={"question": text.strip(), "cognitive_skill": skill if model_verified else ""},
                source="trained_cognitive_router_v11" if model_verified else "local_challenge_guard_v11",
                decision_mode="think",
            )

        # Words such as «سلام» are content inside translation/meaning questions,
        # not conversational greetings.  Route these before greeting signatures.
        if self._LANGUAGE_QUESTION.search(normalized):
            return IntentRoute(
                "knowledge_question", "information", 0.99, 0.78,
                arguments={"question": text.strip()},
                source="language_question_guard_v091",
            )

        # A verb inside a how-to question describes the subject, not permission
        # to execute it.  This guard runs before destructive/action routing.
        if EntityExtractorV7.is_educational(text):
            return IntentRoute(
                "knowledge_question", "information", 0.98, 0.72,
                arguments={"question": text.strip()}, source="educational_polarity_guard_v7",
                structured_entities=tuple(
                    entity.to_dict() for entity in EntityExtractorV7.extract(text, current)
                ),
            )
        if re.fullmatch(
            r"(?:undo|undo\s+last\s+action|آخرین\s+کار(?:م)?\s+رو\s+(?:لغو|برگردون)|"
            r"کار\s+قبلی\s+رو\s+(?:لغو|برگردون)|عملیات\s+قبلی\s+رو\s+برگردون)",
            normalized,
            re.I,
        ):
            return self._tool(
                "undo_last_action", {}, source="undo_router_v7", confirmation=True,
                action="undo", entity_type="action_history",
            )
        if re.search(
            r"(?:حالت\s+تشخیص|عیب.?یابی\s+(?:جارویس|سیستم)|"
            r"جارویس.*(?:مشکل|سلامت).*بررسی|\bdiagnos(?:e|tic)\b|health\s+check)",
            normalized,
            re.I,
        ):
            return self._tool(
                "diagnose_system", {"scope": "jarvis"}, source="diagnose_router_v7",
                action="diagnose", entity_type="system", mode="think",
            )
        forget_match = re.search(
            r"(?:اسم\s+من(?:و|\s+را|\s+رو)?\s+فراموش\s+کن|forget\s+my\s+name)",
            normalized,
            re.I,
        )
        if forget_match:
            return self._tool(
                "forget_memory", {"scope": "semantic", "key": "user_name"},
                source="explicit_forgetting_v7", confirmation=True,
                action="forget", entity_type="memory",
            )
        if re.fullmatch(r"(?:لیست|فهرست|نمایش)\s+(?:مهارت(?:ها)?|skills?)|list\s+skills?", normalized, re.I):
            return self._tool(
                "list_skills", {}, source="extension_discovery_v7",
                action="list", entity_type="skill",
            )
        if re.fullmatch(r"(?:لیست|فهرست|نمایش)\s+(?:پلاگین(?:ها)?|plugins?)|list\s+plugins?", normalized, re.I):
            return self._tool(
                "list_plugins", {}, source="extension_discovery_v7",
                action="list", entity_type="plugin",
            )

        personality_match = re.search(
            r"(?:مود|حالت|شخصیت|mode|personality).*(?:رو|را)?\s*"
            r"(عصبانی|مهربون|مهربان|لوتی|گنگ|خنده\s*دار|حرفه\s*ای|عادی|"
            r"angry|kind|loti|gang|funny|professional|normal)",
            normalized,
            re.I,
        )
        if personality_match:
            aliases = {
                "عصبانی": "Angry", "angry": "Angry", "مهربون": "Kind",
                "مهربان": "Kind", "kind": "Kind", "لوتی": "Loti", "loti": "Loti",
                "گنگ": "Gang", "gang": "Gang", "خنده دار": "Funny", "funny": "Funny",
                "حرفه ای": "Professional", "professional": "Professional",
                "عادی": "Normal", "normal": "Normal",
            }
            key = re.sub(r"\s+", " ", personality_match.group(1).casefold())
            return IntentRoute(
                "set_personality", "conversation", 0.99, 0.8,
                arguments={"personality": aliases[key]}, source="personality_command_v5",
            )

        research_match = re.search(
            r"(?:تحقیق\s*(?:کن|کنید)?|از\s+اینترنت\s+(?:پیدا|بررسی)\s*(?:کن)?|"
            r"بررسی\s+اینترنتی\s*(?:کن)?|\bresearch\b)",
            normalized,
            re.I,
        )
        if research_match:
            query = re.sub(research_match.re, " ", normalized).strip(" :،؟?")
            query = query or text.strip()
            return self._tool(
                "web_research", {"query": query}, source="research_mode_v5",
                mode="research", action="research", entity_type="information",
            )

        if re.search(r"^(?:یه|یک|a)\s+(?:جوک|joke)|(?:جوک|joke).*(?:بگو|tell)", normalized, re.I):
            return IntentRoute("tell_joke", "conversation", 0.99, 0.8, source="creative_router_v6")
        if re.search(
            r"(?:(?:داستان|قصه)\s+کوتاه|short\s+story).*(?:بنویس|بگو|تعریف\s+کن|write|tell)|"
            r"(?:بنویس|بگو|تعریف\s+کن|write|tell)(?:\s+me)?(?:\s+a)?\s+(?:short\s+)?(?:داستان|قصه|story)|"
            r"^(?:یک|یه|a)\s+(?:داستان|قصه|story)",
            normalized, re.I,
        ):
            return IntentRoute("short_story", "conversation", 0.99, 0.8, source="creative_router_v14")
        if re.search(
            r"(?:i\s+feel\s+(?:really\s+)?overwhelmed|i(?:'m|\s+am)\s+overwhelmed|"
            r"need\s+someone\s+to\s+talk\s+to|having\s+a\s+hard\s+day)",
            normalized, re.I,
        ):
            return IntentRoute(
                "support_request", "conversation", 0.99, 0.8,
                arguments={"request": text.strip()}, source="support_router_v14",
            )
        if self._REWRITE_REQUEST.search(normalized):
            return IntentRoute(
                "rewrite_request", "conversation", 0.98, 0.76,
                arguments={"text": text.strip()}, source="persian_fluency_router_v9",
            )
        writing_request = self._WRITING_REQUEST.search(normalized)
        dangerous_tail = bool(
            re.search(r"(?:و\s+بعد|سپس|بعدش|and\s+then).*(?:حذف|پاک\s+کن|delete|shutdown|restart)", normalized, re.I)
        )
        if writing_request and not dangerous_tail:
            return IntentRoute(
                "writing_request", "conversation", 0.98, 0.76,
                arguments={"request": text.strip()}, source="persian_fluency_router_v9",
            )
        if self._ADVICE_REQUEST.search(normalized):
            return IntentRoute(
                "advice_request", "conversation", 0.97, 0.72,
                arguments={"request": text.strip()}, source="persian_fluency_router_v9",
            )

        power_routes = (
            ("shutdown_system", r"(?:سیستم|کامپیوتر|رایانه|ویندوز|pc|computer|system).*(?:خاموش|shutdown)|(?:خاموش|shutdown).*(?:سیستم|کامپیوتر|رایانه|ویندوز|pc|computer|system)"),
            ("restart_system", r"(?:سیستم|کامپیوتر|رایانه|ویندوز|pc|computer|system).*(?:ریستارت|restart|reboot)|(?:ریستارت|restart|reboot).*(?:سیستم|کامپیوتر|رایانه|ویندوز|pc|computer|system)"),
            ("sleep_system", r"(?:سیستم|کامپیوتر|ویندوز|pc|computer).*(?:sleep|خواب)|(?:sleep|بخوابون).*(?:سیستم|کامپیوتر|ویندوز|pc|computer)"),
            ("logoff_system", r"(?:log\s*off|sign\s*out|خروج\s+از\s+ویندوز|لاگ\s*اوت)"),
        )
        for power_intent, pattern in power_routes:
            if re.search(pattern, normalized, re.I):
                return self._tool(
                    power_intent, {}, source="power_control_v6", confirmation=True,
                    action=power_intent.removesuffix("_system"), entity_type="system",
                )

        volume_match = re.search(r"(?:صدا|volume).*(?:درصد|%|روی|بکن|set|to)", normalized, re.I)
        volume_value = PersianNumberParser.find(normalized) if volume_match else None
        if volume_value is not None:
            return self._tool(
                "set_volume", {"percent": max(0, min(100, int(volume_value)))},
                source="system_control_v6", action="change", entity_type="volume",
            )
        if re.search(r"(?:صدا|volume).*(?:زیاد|بیشتر|up)", normalized, re.I):
            return self._tool("volume_up", {"steps": 2}, source="system_control_v6", action="change", entity_type="volume")
        if re.search(r"(?:صدا|volume).*(?:کم|پایین|down)", normalized, re.I):
            return self._tool("volume_down", {"steps": 2}, source="system_control_v6", action="change", entity_type="volume")
        if re.search(r"(?:صدا|volume).*(?:قطع|بی\s*صدا|mute)", normalized, re.I) and not re.search(r"(?:وصل|از\s+بی\s*صدا|unmute)", normalized, re.I):
            return self._tool("mute", {}, source="system_control_v6", action="change", entity_type="volume")
        if re.search(r"(?:صدا|volume).*(?:وصل|از\s+بی\s*صدا|unmute)", normalized, re.I):
            return self._tool("unmute", {}, source="system_control_v6", action="change", entity_type="volume")
        brightness_match = re.search(r"(?:روشنایی|brightness).*(?:درصد|%|روی|بکن|set|to)", normalized, re.I)
        brightness_value = PersianNumberParser.find(normalized) if brightness_match else None
        if brightness_value is not None:
            return self._tool(
                "set_brightness", {"percent": max(0, min(100, int(brightness_value)))},
                source="system_control_v6", action="change", entity_type="display",
            )

        if re.search(r"(?:لیست|فهرست|همه).*(?:پنجره|window)|(?:پنجره|window).*(?:لیست|فهرست)", normalized, re.I):
            return self._tool("enumerate_windows", {}, source="window_router_v6", action="list", entity_type="window")
        if re.search(r"(?:برگرد|برو)\s+عقب|navigate\s+back|browser\s+back", normalized, re.I):
            return self._tool("navigate_back", {}, source="browser_router_v6", action="navigate", entity_type="browser")
        if re.search(r"(?:برو\s+جلو|navigate\s+forward|browser\s+forward)", normalized, re.I):
            return self._tool("navigate_forward", {}, source="browser_router_v6", action="navigate", entity_type="browser")
        if re.search(r"(?:صفحه|پیج|browser|page).*(?:رفرش|تازه|refresh|reload)", normalized, re.I):
            return self._tool("refresh_browser", {}, source="browser_router_v6", action="navigate", entity_type="browser")
        if re.search(r"(?:تب\s+جدید|new\s+tab)", normalized, re.I):
            url = self.entities.explicit_url(text) if self.entities else ""
            return self._tool("new_tab", {"url": url}, source="browser_router_v6", action="open", entity_type="browser_tab")
        if re.search(r"(?:تب|tab).*(?:ببند|close)", normalized, re.I):
            number = re.search(r"\d+", normalized)
            arguments = {"index": max(0, int(number.group()) - 1)} if number else {}
            return self._tool("close_tab", arguments, source="browser_router_v6", action="close", entity_type="browser_tab")
        switch_tab = re.search(r"(?:برو\s+(?:روی|به)|switch\s+to)\s+(?:تب|tab)\s*(\d+)", normalized, re.I)
        if switch_tab:
            return self._tool("switch_tab", {"index": max(0, int(switch_tab.group(1)) - 1)}, source="browser_router_v6", action="focus", entity_type="browser_tab")
        if re.search(r"(?:آدرس|url).*(?:صفحه|تب|page|tab)|current\s+url", normalized, re.I):
            return self._tool("current_url", {}, source="browser_router_v6", action="read", entity_type="url")
        if re.search(r"(?:متن|محتوای).*(?:صفحه|سایت|page).*(?:بخون|read)|read\s+(?:the\s+)?page", normalized, re.I):
            return self._tool("browser_read_page", {}, source="browser_router_v6", action="read", entity_type="web_page")
        if re.search(r"(?:لینک|link).*(?:صفحه|سایت|page).*(?:لیست|نشون|show|list)", normalized, re.I):
            return self._tool("browser_links", {}, source="browser_router_v6", action="list", entity_type="web_page")
        if re.search(r"(?:اولین|اول|first).*(?:نتیجه|result|لینک|link).*(?:باز|open|کلیک|click)", normalized, re.I):
            return self._tool("browser_open_link", {"index": 0}, source="browser_router_v6", action="click", entity_type="web_link")
        click_match = re.search(r"(?:روی\s+)?[\"«']?(.{1,100}?)[\"»']?\s+(?:کلیک\s*(?:کن)?|click)", normalized, re.I)
        if click_match:
            label = click_match.group(1).strip(" .،")
            sensitive = bool(re.search(r"(?:خرید|پرداخت|ارسال|انتشار|ثبت\s+نهایی|buy|pay|submit|publish|send)", label, re.I))
            return self._tool(
                "browser_click", {"label": label, "sensitive": sensitive},
                source="browser_dom_router_v6", confirmation=sensitive,
                action="click", entity_type="web_element",
            )

        if re.search(r"(?:clipboard|کلیپ\s*بورد).*(?:بخون|read|چی|what)", normalized, re.I):
            return self._tool("clipboard_read", {}, source="clipboard_router_v6", action="read", entity_type="clipboard")
        if re.search(r"(?:clipboard|کلیپ\s*بورد).*(?:پاک|خالی|clear)", normalized, re.I):
            return self._tool("clipboard_clear", {}, source="clipboard_router_v6", action="delete", entity_type="clipboard")
        copy_text = re.search(r"(?:این\s+متن|متن\s+[\"«'](.+?)[\"»']|copy\s+[\"'](.+?)[\"']).*(?:کپی|clipboard)", text, re.I)
        if copy_text:
            value = next((item for item in copy_text.groups() if item), text)
            return self._tool("clipboard_write", {"text": value}, source="clipboard_router_v6", action="write", entity_type="clipboard")

        system_routes = (
            ("cpu_info", r"(?:اطلاعات|مشخصات|وضعیت).*(?:cpu|پردازنده)|(?:cpu|پردازنده).*(?:چیه|اطلاعات|مشخصات)"),
            ("gpu_info", r"(?:اطلاعات|مشخصات|وضعیت).*(?:gpu|گرافیک)|(?:gpu|گرافیک).*(?:چیه|اطلاعات|مشخصات)"),
            ("ram_info", r"(?:اطلاعات|مشخصات|وضعیت).*(?:ram|رم)|(?:ram|رم).*(?:چقدره|اطلاعات|مشخصات)"),
            ("battery_info", r"(?:باتری|battery).*(?:وضعیت|چنده|اطلاعات)|(?:وضعیت|اطلاعات).*(?:باتری|battery)"),
            ("network_info", r"(?:شبکه|network).*(?:وضعیت|اطلاعات|مشخصات)|(?:وضعیت|اطلاعات).*(?:شبکه|network)"),
            ("process_info", r"(?:پروسس|process).*(?:اطلاعات|لیست|وضعیت)|(?:لیست|فهرست).*(?:پردازش|process)"),
        )
        for intent_name, pattern in system_routes:
            if re.search(pattern, normalized, re.I):
                return self._tool(intent_name, {}, source="system_information_v6", action="read", entity_type="system")

        if re.search(r"(?:برنامه|نرم\s*افزار|app).*(?:نصب|installed).*(?:لیست|فهرست|show|list)", normalized, re.I):
            return self._tool("list_installed_apps", {}, source="app_index_v6", action="list", entity_type="app")
        find_app_match = re.search(
            r"(?:برنامه|نرم\s*افزار|app)\s+(.{1,80}?)(?=\s+(?:رو|را)?\s*(?:پیدا|find|locate)|$)",
            normalized, re.I,
        )
        if find_app_match and re.search(r"(?:پیدا|find|locate)", normalized, re.I):
            return self._tool(
                "find_installed_app", {"query": find_app_match.group(1).strip(" .،")},
                source="app_index_v6", action="find", entity_type="app",
            )
        if re.search(r"(?:برنامه|اپ|app).*(?:باز|اجرا).*(?:لیست|فهرست)|(?:لیست|فهرست).*(?:برنامه|اپ|app).*(?:باز|اجرا)", normalized, re.I):
            return self._tool("list_running_apps", {}, source="process_router_v6", action="list", entity_type="app")

        if re.search(r"(?:اطلاعات|مشخصات).*(?:فایل|file)", normalized, re.I):
            path_match = self._PATH.search(text)
            last_file = current.get("last_file", "")
            if isinstance(last_file, dict):
                last_file = last_file.get("id", "")
            return self._tool("file_info", {"path": path_match.group(0).strip() if path_match else str(last_file or "")}, source="file_info_v6", action="read", entity_type="file")
        if re.search(r"(?:اطلاعات|مشخصات).*(?:پوشه|فولدر|folder)", normalized, re.I):
            path_match = self._PATH.search(text)
            return self._tool("folder_info", {"path": path_match.group(0).strip() if path_match else str(current.get("last_folder", ""))}, source="file_info_v6", action="read", entity_type="folder")

        name = FactExtractor.user_name(text)
        if name:
            return IntentRoute(
                "set_user_name", "conversation", 1.0, 1.0,
                {"user_name": name}, source="fact_extractor"
            )
        if self._ASK_NAME.search(normalized):
            return IntentRoute("ask_user_name", "conversation", 0.99, 0.8, source="memory_query")
        if self._FINGLISH_GREETING.search(normalized):
            return IntentRoute("greeting", "conversation", 0.97, 0.7, source="language_guard")
        if self._HOW_ARE_YOU_ALT.search(normalized):
            return IntentRoute("how_are_you", "conversation", 0.97, 0.7, source="language_guard")
        if self._LEARNING.search(normalized):
            return IntentRoute("correction_learning", "conversation", 0.98, 0.7, source="learning_gate")
        if self._SETTINGS.search(normalized):
            return self._tool("open_settings", {}, source="ui_control")
        if self._LOW_MOOD.search(normalized):
            return IntentRoute("user_low_mood", "conversation", 0.99, 0.8, source="mood_guard")
        if self._SMALLTALK.search(normalized):
            return IntentRoute("smalltalk", "conversation", 0.97, 0.7, source="smalltalk_guard")
        if self._PRIVACY.search(normalized):
            return IntentRoute("offline_privacy", "conversation", 0.98, 0.75, source="privacy_guard")
        if self._PROJECT_KIND.search(normalized):
            return IntentRoute("project_kind", "conversation", 0.97, 0.7, source="project_guard")
        if self._COMPLEX_HINT.search(normalized):
            return IntentRoute(
                "complex_question", "think", 0.96, 0.65,
                arguments={"question": text.strip()}, source="complexity_guard",
                decision_mode="think",
            )
        if self._VAGUE_UNKNOWN.search(normalized):
            return IntentRoute(
                "unknown", "unknown", 0.92, 0.65,
                source="vague_input_guard", unknown_reason="insufficient_subject",
            )
        if re.search(r"(?:dangerous\s+command|command\s+خطرناک|دستور\s+خطرناک)", normalized, re.I):
            return IntentRoute(
                "dangerous_request", "tool", 0.99, 0.9,
                arguments={"request": text.strip()}, source="safety_gate",
                requires_confirmation=True,
            )
        if self._CLEAR_CHAT.search(normalized):
            return self._tool(
                "clear_conversation", {}, source="conversation_control", confirmation=True
            )

        workflow = FileWorkflowParser.parse(text)
        if workflow is not None:
            return IntentRoute(
                "file_selection_workflow", "tool", 0.98, 0.74,
                arguments={"workflow": workflow.to_dict()},
                source="compositional_planner_nlu_v7", decision_mode="think",
                action=workflow.final_action.removesuffix("_file"), entity_type="file",
                structured_entities=tuple(
                    entity.to_dict() for entity in EntityExtractorV7.extract(text, current)
                ),
                dry_run=EntityExtractorV7.is_dry_run(text),
            )

        if allow_multi:
            segments = TaskSegmenter.split(text)
            if len(segments) >= 2:
                return IntentRoute(
                    "multi_step_task", "tool", 0.96, 0.55,
                    arguments={"segments": list(segments)}, source="task_segmenter_v4", decision_mode="think",
                )

        if self._DATE.search(normalized):
            return self._tool("date_time", {}, source="datetime_gate")
        if self._SYSTEM.search(normalized):
            if re.search(r"(?:خوبه|مناسب|کافیه|اجرا|برای\s+این\s+پروژه|good\s+enough|suitable|can\s+it\s+run)", normalized, re.I):
                return IntentRoute(
                    "system_assessment", "think", 0.97, 0.7,
                    arguments={"question": text.strip(), "requires_system_info": True},
                    source="tool_aware_reasoning", decision_mode="think",
                    action="inspect", entity_type="system",
                )
            return self._tool("system_info", {}, source="system_gate")

        semantic_route = self._semantic_action_route(text, normalized, current)
        if semantic_route:
            return semantic_route

        if CalculatorTool.looks_like_calculation(text):
            return self._tool(
                "calculator", {"expression": CalculatorTool.extract_expression(text)},
                source="calculator_gate",
            )

        path_match = self._PATH.search(text)
        explicit_path = path_match.group(0).strip() if path_match else ""
        if self._ZIP.search(normalized) and (self._READ.search(normalized) or "داخل" in normalized or "list" in normalized):
            return self._tool("zip_inspect", {"path": explicit_path}, source="file_router")
        standalone_file = re.search(r"[\w.-]+\.[A-Za-z0-9]{1,12}", normalized)
        if self._FIND.search(normalized) and standalone_file:
            folder = explicit_path or str(current.get("last_folder", ""))
            return self._tool(
                "find_file", {"folder": folder, "query": standalone_file.group(0)},
                source="file_router",
            )
        if self._FOLDER.search(normalized):
            folder = explicit_path or str(current.get("last_folder", ""))
            if self._FIND.search(normalized):
                file_match = re.search(r"[\w.-]+\.[A-Za-z0-9]{1,12}", normalized)
                query = file_match.group(0) if file_match else normalized
                return self._tool("find_file", {"folder": folder, "query": query}, source="file_router")
            if self._READ.search(normalized) or re.search(r"(?:لیست|داخل|چه فایل|what files|list|contents)", normalized):
                return self._tool("list_folder", {"path": folder}, source="file_router")
            if self._OPEN_HINT.search(normalized):
                return self._tool("open_folder", {"path": folder}, source="file_router")
        if self._READ.search(normalized) and re.search(r"(?:فایل|پیوست|attachment|markdown|readme|log|\.\w{1,8}\b|file)", normalized):
            resolved_path = explicit_path
            if not resolved_path:
                file_match = re.search(r"[\w.-]+\.[A-Za-z0-9]{1,12}", text)
                last_folder = str(current.get("last_folder", ""))
                if file_match and last_folder:
                    resolved_path = str(Path(last_folder) / file_match.group(0))
            if not resolved_path:
                last_file = current.get("last_file", "")
                if isinstance(last_file, dict):
                    last_file = last_file.get("id", "")
                resolved_path = str(last_file or "")
            return self._tool("read_file", {"path": resolved_path}, source="file_router")

        entity_route = self._entity_route(text, normalized, current)
        if entity_route:
            return entity_route

        assessment = DecisionEngine.assess(text)
        if self._IMPOSSIBLE_FUTURE.search(normalized):
            return IntentRoute(
                "unknown", "unknown", 0.96, 0.7, source="future_claim_guard",
                unknown_reason="unverifiable_future_claim",
            )
        if self._DANGEROUS.search(normalized):
            return IntentRoute(
                "dangerous_request", "tool", 0.95, 0.7,
                arguments={"request": text.strip()}, source="safety_gate",
                requires_confirmation=True,
            )
        if assessment.needs_fresh_information:
            return self._tool(
                "web_search", {"query": text.strip()}, confidence=assessment.confidence,
                source="freshness_detector", mode="search",
            )
        prediction: IntentPrediction | None = self.brain.classify(text) if self.brain else None
        signature = self._signature_match(normalized)
        if signature:
            open_question = bool(re.search(
                r"(?:\b(?:what|why|how|explain|compare|difference|define)\b|"
                r"(?:چیست|چیه|یعنی\s+چی|چرا|توضیح\s+بده|فرق|تفاوت|مقایسه))",
                normalized,
                re.I,
            )) or (
                text.strip().endswith(("?", "؟")) and len(normalized.split()) >= 3
            )
            conversational_questions = {
                "ask_name", "ask_assistant_name", "ask_user_name", "how_are_you",
                "ask_capabilities", "capabilities", "self_intro", "greeting",
            }
            if open_question and signature.intent not in conversational_questions:
                return IntentRoute(
                    "complex_question" if assessment.is_complex else "knowledge_question",
                    "think" if assessment.is_complex else "information",
                    0.94, 0.52,
                    arguments={"question": text.strip()},
                    source="question_polarity_guard_v071",
                    decision_mode="think" if assessment.is_complex else "fast",
                )
            neural_score = 0.0
            alternatives: tuple[tuple[str, float], ...] = ()
            if prediction:
                alternatives = prediction.alternatives
                neural_score = next((score for intent, score in alternatives if intent == signature.intent), 0.0)
            confidence = min(0.99, signature.boost + neural_score * 0.1)
            mode = "think" if assessment.is_complex else "fast"
            canonical_intent = self._CANONICAL_INTENTS.get(signature.intent, signature.intent)
            canonical_tools = {
                "date_time", "read_file", "zip_inspect", "web_search",
                "open_default_browser",
            }
            route_type = "tool" if canonical_intent in canonical_tools else signature.route_type
            route_type = "think" if assessment.is_complex and route_type in {"information", "unknown"} else route_type
            intent = "complex_question" if route_type == "think" else canonical_intent
            arguments: dict[str, Any] = {}
            if intent == "web_search":
                arguments = {"query": text.strip()}
            return IntentRoute(
                intent, route_type, confidence,
                max(0.1, confidence - (prediction.confidence if prediction else 0.0)),
                arguments=arguments, alternatives=alternatives,
                source="hybrid_signature", decision_mode=mode,
            )
        if assessment.is_complex:
            return IntentRoute(
                "complex_question", "think", assessment.confidence, 0.35,
                arguments={"question": text.strip()}, source="complexity_detector", decision_mode="think",
            )
        if re.search(
            r"(?:\b(?:what|why|how|explain|compare|difference|define)\b|"
            r"(?:چیست|چیه|یعنی\s+چی|چرا|توضیح\s+بده|فرق|تفاوت|مقایسه))",
            normalized,
            re.I,
        ) or (text.strip().endswith(("?", "؟")) and len(normalized.split()) >= 3):
            return IntentRoute(
                "knowledge_question", "information", 0.91, 0.48,
                arguments={"question": text.strip()}, source="open_domain_question_router_v7",
                decision_mode="fast",
            )
        if prediction is None:
            return IntentRoute(
                "unknown", "unknown", 0.0, 0.0,
                source="fallback", unknown_reason="brain_unavailable"
            )
        return IntentRoute(
            prediction.intent, prediction.route_type, prediction.confidence,
            prediction.margin, alternatives=prediction.alternatives,
            source="neural_hybrid", unknown_reason=prediction.unknown_reason,
        )
