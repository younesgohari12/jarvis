from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from jarvis.utils.text import normalize_text


@dataclass(frozen=True, slots=True)
class FluentResponse:
    text: str
    label: str
    confidence: float
    source: str
    checks: tuple[str, ...]


class PersianFluencyEngine:
    """Small trained intent selector plus conservative Persian response composer."""

    FORMAT = "jarvis-persian-fluency-v1"
    _REQUEST_HINT = re.compile(
        r"(?:بنویس|می.?نویسی|نگارش|بگو|بساز|درست\s+کن|آماده\s+(?:کن|می.?کنی)|"
        r"بازنویسی|روان.?تر|بهترش\s+کن|پیشنهاد|"
        r"چطور\s+شروع|چه\s+کار\s+کنم|چی\s+کار\s+کنم|برنامه.?ریزی|"
        r"کپشن|پیام|ایمیل|نامه|داستان|قصه|متن|نوشته|چند\s+خط)",
        re.I,
    )
    _FORCED_LABELS = {
        "writing_request": "text",
        "rewrite_request": "rewrite",
        "advice_request": "advice",
        "short_story": "story",
        "user_low_mood": "support",
        "support_request": "support",
    }

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.ready = False
        self.version = ""
        self._idf: dict[str, float] = {}
        self._centroids: dict[str, dict[str, float]] = {}
        self._examples: tuple[dict[str, str], ...] = ()
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("format") != self.FORMAT:
            return
        idf = payload.get("idf", {})
        centroids = payload.get("centroids", {})
        examples = payload.get("examples", [])
        if not isinstance(idf, dict) or not isinstance(centroids, dict):
            return
        self._idf = {str(key): float(value) for key, value in idf.items()}
        self._centroids = {
            str(label): {str(key): float(value) for key, value in vector.items()}
            for label, vector in centroids.items() if isinstance(vector, dict)
        }
        self._examples = tuple(
            {
                "prompt": str(row.get("prompt", "")),
                "label": str(row.get("label", "")),
                "response": str(row.get("response", "")),
            }
            for row in examples if isinstance(row, dict)
        )
        self.version = str(payload.get("version", ""))
        self.ready = bool(self._idf and self._centroids and self._examples)

    @staticmethod
    def features(text: str) -> Counter[str]:
        normalized = normalize_text(text)
        words = re.findall(r"[a-z0-9\u0600-\u06ff]+", normalized, re.I)
        values: Counter[str] = Counter(f"w:{word}" for word in words)
        compact = re.sub(r"\s+", " ", normalized).strip()
        padded = f"  {compact}  "
        values.update(
            f"c:{padded[index:index + 3]}"
            for index in range(max(0, len(padded) - 2))
        )
        return values

    @classmethod
    def weighted_vector(
        cls, text: str, idf: dict[str, float], *, allowed: set[str] | None = None,
    ) -> dict[str, float]:
        counts = cls.features(text)
        vector = {
            key: (1.0 + math.log(value)) * idf.get(key, 1.0)
            for key, value in counts.items()
            if allowed is None or key in allowed
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {key: value / norm for key, value in vector.items()}

    @staticmethod
    def similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(key, 0.0) for key, value in left.items())

    def predict(self, text: str) -> tuple[str, float]:
        if not self.ready:
            return "", 0.0
        allowed = set(self._idf)
        vector = self.weighted_vector(text, self._idf, allowed=allowed)
        ranked = sorted(
            (
                (self.similarity(vector, centroid), label)
                for label, centroid in self._centroids.items()
            ),
            reverse=True,
        )
        if not ranked:
            return "", 0.0
        best_score, best_label = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        confidence = max(0.0, min(0.99, best_score * 0.72 + max(0.0, best_score - second) * 0.9))
        return best_label, round(confidence, 4)

    @staticmethod
    def _clean_topic(value: str) -> str:
        topic = re.sub(r"[؟?!]+$", "", value).strip(" \t\n،,.:؛;«»\"'")
        topic = re.sub(
            r"\s+(?:یک|یه)?\s*(?:متن|نوشته|پیام|کپشن|داستان|قصه|پاراگراف|نامه|ایمیل)"
            r"(?:\s+(?:زیبا|کوتاه|بلند|روان|رسمی|صمیمی|جذاب|خلاقانه|محترمانه))*"
            r"\s*(?:بنویس|می.?نویسی|بگو|بساز|تعریف\s+کن|درست\s+کن|آماده\s+(?:کن|می.?کنی))?$",
            "",
            topic,
            flags=re.I,
        )
        topic = re.sub(
            r"\s+(?:بنویس|می.?نویسی|بگو|بساز|تعریف\s+کن|درست\s+کن|آماده\s+(?:کن|می.?کنی))$",
            "", topic, flags=re.I,
        )
        return re.sub(r"\s+", " ", topic).strip()[:120]

    @classmethod
    def _topic(cls, text: str) -> str:
        normalized = normalize_text(text)
        for pattern in (
            r"(?:درباره|در\s+مورد|راجع\s+به)\s+(.+)",
            r"(?:برای|با\s+موضوع)\s+(.+?)\s+(?:بنویس|می.?نویسی|بگو|بساز|تعریف\s+کن|درست\s+کن|آماده\s+(?:کن|می.?کنی))$",
            r"(?:معرفی|موضوع)\s+(.+?)\s+(?:بنویس|می.?نویسی|بگو|بساز|تعریف\s+کن|درست\s+کن|آماده\s+(?:کن|می.?کنی))$",
        ):
            match = re.search(pattern, normalized, re.I)
            if match:
                topic = cls._clean_topic(match.group(1))
                if topic:
                    return topic
        cleaned = re.sub(
            r"^(?:لطفاً\s+)?(?:یک|یه)?\s*(?:متن|نوشته|پیام|کپشن|داستان|قصه|پاراگراف|نامه|ایمیل)"
            r"(?:\s+(?:زیبا|کوتاه|بلند|روان|رسمی|صمیمی|جذاب|خلاقانه|محترمانه))*\s*",
            "",
            normalized,
            flags=re.I,
        )
        cleaned = re.sub(
            r"\s*(?:بنویس|می.?نویسی|بگو|بساز|تعریف\s+کن|درست\s+کن|آماده\s+(?:کن|می.?کنی))$",
            "", cleaned, flags=re.I,
        )
        return cls._clean_topic(cleaned) or "این موضوع"

    @staticmethod
    def _writing_kind(text: str, predicted: str) -> str:
        value = normalize_text(text)
        if re.search(r"(?:بازنویسی|بازنویسیش|روان.?تر|بهترش\s+کن|ویرایش)", value, re.I):
            return "rewrite"
        if re.search(r"(?:پیام|ایمیل|نامه).*(?:رسمی|محترمانه|مودبانه|اداری)|(?:رسمی|محترمانه|مودبانه|اداری).*(?:پیام|ایمیل|نامه|متن)", value, re.I):
            return "formal_message"
        if re.search(r"(?:کپشن|پست\s+(?:اینستاگرام|شبکه))", value, re.I):
            return "caption"
        if "داستان" in value or "قصه" in value:
            return "story"
        return predicted if predicted in {
            "text", "formal_message", "caption", "story", "advice", "support", "rewrite"
        } else "text"

    def _nearest_response(self, text: str, label: str) -> tuple[str, float]:
        vector = self.weighted_vector(text, self._idf, allowed=set(self._idf))
        ranked: list[tuple[float, str]] = []
        for row in self._examples:
            if row["label"] != label or not row["response"]:
                continue
            candidate = self.weighted_vector(row["prompt"], self._idf, allowed=set(self._idf))
            ranked.append((self.similarity(vector, candidate), row["response"]))
        if not ranked:
            return "", 0.0
        score, response = max(ranked, key=lambda item: item[0])
        return response, score

    @staticmethod
    def _motivational_text(topic: str) -> str:
        normalized = normalize_text(topic)
        if re.search(r"(?:تلاش|پشتکار|ادامه\s+دادن)", normalized):
            return (
                "پیشرفت همیشه با قدم‌های بزرگ آغاز نمی‌شود؛ گاهی فقط کافی است یک روز دیگر "
                "ادامه بدهی. تلاش، مسیر را می‌سازد و پشتکار کمک می‌کند از روزهایی عبور کنی که "
                "نتیجه هنوز دیده نمی‌شود. موفقیت اغلب همان لحظه‌ای شکل می‌گیرد که تصمیم می‌گیری "
                "کمی بیشتر دوام بیاوری."
            )
        if "امید" in normalized:
            return (
                "امید یعنی حتی وقتی تمام مسیر روشن نیست، به اندازهٔ قدم بعدی نور داشته باشی. "
                "گاهی همین روشنایی کوچک کافی است تا دوباره حرکت کنیم، راه تازه‌ای ببینیم و باور کنیم "
                "که هیچ شبِ طولانی‌ای همیشگی نیست."
            )
        if re.search(r"(?:شروع|آغاز)", normalized):
            return (
                "شروع‌کردن همیشه به آمادگی کامل نیاز ندارد؛ بیشتر وقت‌ها، آمادگی در دل حرکت ساخته "
                "می‌شود. یک قدم کوچک اما واقعی بردار؛ مسیر بعد از همان قدم، واضح‌تر از قبل خواهد شد."
            )
        return (
            f"{topic} زمانی معنا پیدا می‌کند که از یک فکر زیبا به رفتاری واقعی تبدیل شود. "
            "تغییرهای ماندگار معمولاً آرام و پیوسته شکل می‌گیرند؛ با انتخاب‌های کوچک، توجه به مسیر "
            "و شجاعتِ ادامه‌دادن حتی وقتی نتیجه فوری نیست."
        )

    @staticmethod
    def _formal_message(text: str, topic: str) -> str:
        normalized = normalize_text(text)
        if "سفارش" in normalized and re.search(r"(?:فردا|روز\s+آینده)", normalized):
            if re.search(r"(?:می.?رسد|می.?رسه|تحویل|دریافت)", normalized):
                return (
                    "سلام و وقت بخیر،\n\n"
                    "طبق برنامه، سفارش شما فردا تحویل خواهد شد. در صورت ایجاد هرگونه تغییر در "
                    "زمان تحویل، اطلاعات تازه در سریع‌ترین زمان برایتان ارسال می‌شود.\n\n"
                    "از همراهی شما سپاسگزاریم."
                )
            return (
                "سلام و وقت بخیر،\n\n"
                "سفارش شما آماده شده است و فردا ارسال خواهد شد. پس از تحویل مرسوله به شرکت "
                "حمل‌ونقل، اطلاعات پیگیری نیز برای شما فرستاده می‌شود.\n\n"
                "از همراهی و شکیبایی شما سپاسگزاریم."
            )
        if re.search(r"(?:تاخیر|تأخیر|دیر)", normalized):
            return (
                "سلام و وقت بخیر،\n\n"
                "به‌دلیل تأخیر ایجادشده صمیمانه عذرخواهی می‌کنیم. موضوع در حال پیگیری است و نتیجه "
                "در اولین فرصت به اطلاع شما خواهد رسید.\n\n"
                "از صبوری و همراهی شما سپاسگزاریم."
            )
        return (
            "سلام و وقت بخیر،\n\n"
            f"در ارتباط با {topic}، خواستم توضیحات لازم را با شما در میان بگذارم. موضوع با دقت "
            "در حال بررسی است و به‌محض نهایی‌شدن، نتیجه را اطلاع خواهیم داد.\n\n"
            "از توجه و همراهی شما سپاسگزارم."
        )

    @staticmethod
    def _caption(topic: str) -> str:
        normalized = normalize_text(topic)
        if "پروژه" in normalized:
            return (
                "هر پروژهٔ تازه، از یک ایدهٔ کوچک و جسارتِ شروع‌کردن به‌وجود می‌آید. ✨\n\n"
                "امروز خوشحالیم بخشی از مسیری را معرفی کنیم که با فکر، آزمون و تلاش ساخته شده است. "
                "این تازه آغاز راه است و مشتاقیم قدم‌های بعدی را هم با شما به اشتراک بگذاریم."
            )
        return (
            f"{topic}؛ روایتی از یک شروع تازه و قدم‌هایی که با دقت برداشته شده‌اند. ✨\n\n"
            "هنوز در ابتدای مسیر هستیم، اما برای ساختن ادامه‌ای بهتر، انگیزه و ایده‌های زیادی داریم."
        )

    @staticmethod
    def _story(topic: str) -> str:
        normalized = normalize_text(topic)
        if normalized == "این موضوع":
            return (
                "نیمه‌شب، جارویس متوجه شد چراغ اتاق کار هنوز روشن است. روی میز، یادداشتی نیمه‌کاره "
                "کنار یک فنجان سرد مانده بود: «فردا دوباره امتحان می‌کنم.» پنجره را بست و یادداشت را "
                "همان‌جا نگه داشت؛ بعضی داستان‌ها با یک پیروزی بزرگ تمام نمی‌شوند، با تصمیمی آرام "
                "برای ادامه‌دادن تازه آغاز می‌شوند."
            )
        if "امید" in normalized:
            return (
                "باران سه روز بود که بند نمی‌آمد. سارا هر صبح گلدان خشک کنار پنجره را نگاه می‌کرد "
                "و بااین‌حال، کمی آب پای آن می‌ریخت. صبح چهارم، ابرها کنار رفتند و جوانه‌ای سبز از "
                "میان خاک پیدا شد. سارا لبخند زد؛ نه فقط برای جوانه، برای این‌که فهمید امید گاهی "
                "همان مراقبت کوچکی است که پیش از دیدن نتیجه ادامه می‌دهیم."
            )
        if re.search(r"(?:تلاش|پشتکار)", normalized):
            return (
                "آرمان هر غروب یک خط کج روی کاغذ می‌کشید و آن را دور می‌انداخت. روز صدم، وقتی "
                "دفتر قدیمی‌اش را ورق زد، دید خط‌های کج آرام‌آرام به طرحی دقیق تبدیل شده‌اند. آن روز "
                "فهمید استعداد همیشه ناگهانی پیدا نمی‌شود؛ گاهی از جمع‌شدن صد تلاش کوچک ساخته می‌شود."
            )
        if "شجاعت" in normalized:
            return (
                "رها از روی صحنه به جمعیت نگاه کرد و صدایش لرزید. یک نفس عمیق کشید و جملهٔ اول را "
                "گفت. تشویق پایان برنامه شیرین بود، اما خودش می‌دانست شجاعت واقعی همان لحظه‌ای بود "
                "که با وجود ترس، تصمیم گرفت آغاز کند."
            )
        return (
            f"غروب آرامی بود که نیما نشانه‌ای از {topic} را در جایی دید که انتظارش را نداشت. "
            "به‌جای عبورکردن، چند لحظه ایستاد و با دقت نگاه کرد. همان مکث کوتاه تصمیمش را عوض کرد؛ "
            "گاهی یک اتفاق کوچک کافی است تا مسیر یک روز معمولی، به آغاز داستانی تازه تبدیل شود."
        )

    @staticmethod
    def _advice(text: str) -> str:
        normalized = normalize_text(text)
        if re.search(r"(?:تمرکز|حواس.{0,12}پرت|برنامه.?ریزی|امروز|کارها)", normalized):
            return (
                "برای امروز دنبال برنامهٔ بی‌نقص نباش؛ فقط شروع را آسان کن:\n\n"
                "1. یک کار اصلی انتخاب کن که انجام‌شدنش بیشترین اثر را دارد.\n"
                "2. آن را به یک قدم ۲۵دقیقه‌ای تبدیل کن و اعلان‌ها را کنار بگذار.\n"
                "3. بعد از ۲۵ دقیقه، پنج دقیقه استراحت کن و فقط سپس دربارهٔ ادامه تصمیم بگیر.\n\n"
                "اگر انرژی‌ات کم است، هدف امروز «حرکت» باشد، نه تمام‌کردن همه‌چیز."
            )
        if re.search(r"(?:شروع|گیر\s+کرد|نمی.?دونم)", normalized):
            return (
                "اول مسئله را به کوچک‌ترین خروجی قابل‌مشاهده تبدیل کن. سپس فقط ده دقیقه برای ساختن "
                "نسخهٔ اولیه وقت بگذار؛ نه برای کامل‌کردنش. وقتی چیزی هرچند ناقص جلوی چشم باشد، "
                "تصمیم‌گرفتن دربارهٔ قدم بعدی بسیار ساده‌تر می‌شود."
            )
        return (
            "پیشنهاد من این است که ابتدا هدفت را در یک جملهٔ روشن بنویسی، بعد مهم‌ترین مانع را مشخص "
            "کنی و برای آن فقط یک اقدام کوچک انتخاب کنی. راه‌حل خوب معمولاً از یک قدم قابل‌آزمایش "
            "شروع می‌شود، نه از یک برنامهٔ طولانی و مبهم."
        )

    @staticmethod
    def _support() -> str:
        return (
            "متأسفم که امروز حالت خوب نیست. لازم نیست همین حالا همه‌چیز را حل کنی یا حتی دقیق توضیح "
            "بدهی؛ من اینجام و بدون قضاوت گوش می‌دهم. اگر دوست داری، فقط بگو بیشتر خستگی است، "
            "فشار کارهاست یا اتفاق مشخصی ذهنت را درگیر کرده؟"
        )

    @staticmethod
    def _rewrite(text: str) -> str:
        # v15: extract embedded source semantically; a colon is optional.
        patterns = (
            r"(?:بازنویسی(?:ش)?\s+کن|روان.?تر(?:ش)?\s+(?:کن|بنویس)|رسمی.?تر(?:ش)?\s+(?:کن|بنویس)|"
            r"حرفه.?ای.?تر(?:ش)?\s+(?:کن|بنویس)|ویرایش\s+کن|اصلاح\s+کن|بهترش\s+کن)\s*[:：-]?\s*[«\"']?(.+?)[»\"']?$",
            r"(?:این\s+(?:جمله|متن|پیام)\s+را|این\s+رو)\s+(?:روان.?تر|رسمی.?تر|حرفه.?ای.?تر|بهتر)\s*(?:کن|بنویس)\s*[:：-]?\s*[«\"']?(.+?)[»\"']?$",
            r"(?:[:：]\s*|متن\s+(?:زیر|این)\s*)(.+)$",
        )
        source = ""
        for pattern in patterns:
            match = re.search(pattern, text, re.I | re.S)
            if match:
                source = match.group(1).strip(" «»\"'")
                if source:
                    break
        if not source:
            return (
                "حتماً. متن را در همین پیام قرار بده؛ مثلاً «رسمی‌تر کن: ...». "
                "اگر لحن خاصی می‌خواهی—رسمی، صمیمی، کوتاه یا تبلیغاتی—همان ابتدا مشخص کن."
            )
        value = normalize_text(source)
        formal_request = bool(re.search(r"(?:رسمی.?تر|حرفه.?ای.?تر|formal|professional)", normalize_text(text), re.I))
        replacements = {
            "می باشد": "است", "میباشد": "است", "می گردد": "می‌شود",
            "در رابطه با": "دربارهٔ", "به اطلاع می رساند": "به اطلاع می‌رسانیم",
            "با تشکر از شما": "از شما سپاسگزاریم", "میخوام": "می‌خواهم",
            "می خوام": "می‌خواهم", "میخام": "می‌خواهم", "میخایم": "می‌خواهیم",
            "سفارشتون": "سفارش شما", "تموم": "تمام", "بگم": "بگویم",
            "بدونم": "بدانم", "میدونم": "می‌دانم", "نمیدونم": "نمی‌دانم",
            "گزارشو": "گزارش را", "فایلو": "فایل را", "متنو": "متن را",
            "برام": "برای من", "برات": "برای شما", "براتون": "برای شما",
            "کنین": "کنید", "بدین": "دهید", "بگین": "بگویید", "بفرستین": "ارسال کنید",
            "واسه": "برای", "چون که": "زیرا", "یه": "یک", "نمیشه": "نمی‌شود",
            "میشه": "می‌شود", "می تونم": "می‌توانم", "میتونم": "می‌توانم",
            "لطفا": "لطفاً", "خواهشا": "خواهش می‌کنم",
            "نمیام": "نمی‌آیم", "نمیایم": "نمی‌آییم", "چون": "زیرا",
        }
        for old, new_value in replacements.items():
            value = value.replace(old, new_value)
        value = re.sub(
            r"^(.+?)\s+دارای\s+کیفیت\s+(?:خیلی\s+خوب|بسیار\s+خوب)\s+است$",
            r"\1 کیفیت بسیار خوبی دارد",
            value,
        )
        value = re.sub(r"\s+رو\s+", " را ", value)
        if formal_request:
            # Conservative register upgrade for common colloquial imperatives.
            # These transformations preserve the requested meaning rather than
            # inventing new content.
            value = re.sub(r"(?<!\S)زود(?!\S)", "در اسرع وقت", value)
            value = re.sub(r"(?<!\S)سریع(?!\S)", "در اسرع وقت", value)
            value = re.sub(r"(?<!\S)بفرست(?:ید)?(?!\S)", "ارسال کنید", value)
            value = re.sub(r"(?<!\S)بگو(?!\S)", "بیان کنید", value)
            value = re.sub(r"(?<!\S)بده(?!\S)", "ارائه کنید", value)
            value = re.sub(r"(?<!\S)چک\s+کن(?:ید)?(?!\S)", "بررسی کنید", value)
        value = re.sub(r"\s*[,،]\s*", "، ", value)
        value = re.sub(r"\s+", " ", value).strip()
        if value and value[-1] not in ".!؟":
            value += "."
        title = "نسخهٔ رسمی‌تر" if formal_request else "نسخهٔ روان‌تر"
        return f"{title}:\n\n{value}"

    @staticmethod
    def _quality(text: str) -> tuple[bool, tuple[str, ...]]:
        value = text.strip()
        checks: list[str] = []
        if 45 <= len(value) <= 1800:
            checks.append("bounded_length")
        else:
            return False, ("invalid_length",)
        if not re.search(r"<\/?(?:tool|assistant|user|system)", value, re.I):
            checks.append("no_control_tokens")
        else:
            return False, ("control_token",)
        letters = re.findall(r"[A-Za-z\u0600-\u06ff]", value)
        persian = re.findall(r"[\u0600-\u06ff]", value)
        if letters and len(persian) / len(letters) >= 0.72:
            checks.append("persian_consistency")
        else:
            return False, ("language_mismatch",)
        sentences = [part.strip() for part in re.split(r"[.!؟\n]+", value) if part.strip()]
        normalized_sentences = [normalize_text(part) for part in sentences]
        if len(normalized_sentences) == len(set(normalized_sentences)):
            checks.append("no_sentence_repetition")
        else:
            return False, ("sentence_repetition",)
        return True, tuple(checks)

    def compose(self, text: str, routed_intent: str, language: str) -> FluentResponse | None:
        if language != "fa" or not self.ready:
            return None
        predicted, predicted_confidence = self.predict(text)
        forced = self._FORCED_LABELS.get(routed_intent, "")
        if not forced and routed_intent not in {"unknown", "smalltalk"}:
            return None
        if not forced and (not self._REQUEST_HINT.search(normalize_text(text)) or predicted_confidence < 0.24):
            return None
        label = self._writing_kind(text, forced or predicted)
        topic = self._topic(text)
        if label == "formal_message":
            output = self._formal_message(text, topic)
        elif label == "caption":
            output = self._caption(topic)
        elif label == "story":
            output = self._story(topic)
        elif label == "advice":
            output = self._advice(text)
        elif label == "support":
            output = self._support()
        elif label == "rewrite":
            output = self._rewrite(text)
        else:
            output = self._motivational_text(topic)
        accepted, checks = self._quality(output)
        if not accepted:
            fallback, similarity = self._nearest_response(text, label)
            accepted, checks = self._quality(fallback) if fallback else (False, ())
            if not accepted:
                return None
            output = fallback
            predicted_confidence = max(predicted_confidence, similarity)
        confidence = max(0.72 if forced else 0.64, min(0.96, predicted_confidence + 0.18))
        return FluentResponse(
            output, label, round(confidence, 4), "trained_persian_fluency", checks,
        )

    @staticmethod
    def polish_grounded(text: str, query: str, language: str) -> str:
        if language != "fa" or not re.search(r"(?:ساده|روان|قابل\s+فهم)", normalize_text(query)):
            return text
        value = text.strip()
        if not value or value.startswith("به زبان ساده"):
            return value
        return f"به زبان ساده، {value[0].lower() + value[1:] if value[:1].isascii() else value}"


def train_centroids(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    examples = [
        {
            "prompt": str(row.get("prompt", "")).strip(),
            "label": str(row.get("label", "")).strip(),
            "response": str(row.get("response", "")).strip(),
        }
        for row in rows
        if str(row.get("prompt", "")).strip() and str(row.get("label", "")).strip()
    ]
    document_frequency: Counter[str] = Counter()
    for row in examples:
        document_frequency.update(PersianFluencyEngine.features(row["prompt"]).keys())
    count = max(1, len(examples))
    idf = {
        key: round(math.log((count + 1) / (frequency + 1)) + 1.0, 8)
        for key, frequency in document_frequency.items()
    }
    label_vectors: dict[str, Counter[str]] = {}
    label_counts: Counter[str] = Counter()
    for row in examples:
        vector = PersianFluencyEngine.weighted_vector(row["prompt"], idf)
        label_vectors.setdefault(row["label"], Counter()).update(vector)
        label_counts[row["label"]] += 1
    centroids: dict[str, dict[str, float]] = {}
    retained: set[str] = set()
    for label, values in label_vectors.items():
        averaged = {key: value / label_counts[label] for key, value in values.items()}
        top = dict(sorted(averaged.items(), key=lambda item: item[1], reverse=True)[:420])
        norm = math.sqrt(sum(value * value for value in top.values())) or 1.0
        centroid = {key: round(value / norm, 8) for key, value in top.items()}
        centroids[label] = centroid
        retained.update(centroid)
    return {
        "format": PersianFluencyEngine.FORMAT,
        "version": "0.9.0",
        "pretrained_source": None,
        "training_method": "word_and_character_tfidf_centroid",
        "training_examples": len(examples),
        "labels": dict(sorted(label_counts.items())),
        "idf": {key: idf[key] for key in sorted(retained)},
        "centroids": dict(sorted(centroids.items())),
        "examples": examples,
    }
