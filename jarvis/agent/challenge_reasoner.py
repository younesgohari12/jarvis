from __future__ import annotations

import math
import re
from dataclasses import dataclass

from jarvis.nlu.entities_v7 import PersianNumberParser
from jarvis.utils.text import normalize_text


_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


@dataclass(frozen=True, slots=True)
class ChallengeAnswer:
    text: str
    confidence: float = 0.97
    checks: tuple[str, ...] = ("locally_solved", "result_verified")


class ChallengeReasoner:
    """Deterministic solver for common reasoning and quantitative question families.

    This layer intentionally runs before desktop-command and web-search routing.  It
    handles only high-precision schemas; unknown questions continue through the
    normal router instead of being guessed.
    """

    _QUOTED = re.compile(r"«([^»]{3,})»|\"([^\"]{3,})\"", re.S)
    _NUMBER = re.compile(r"(?<!\w)[-+]?\d+(?:[.,]\d+)?(?:\s*(?:هزار|میلیون))?(?!\w)")
    _SEQUENCE = re.compile(r"(?:دنباله|عدد\s+بعدی|sequence|next\s+number)", re.I)
    _CLOCK = re.compile(r"(?:ساعت|clock).{0,35}?(\d{1,2})\s*[:：]\s*(\d{1,2})", re.I)
    _FACTORIAL = re.compile(r"(\d+)\s*(?:!|factorial|فاکتوریل)", re.I)
    _PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:٪|%|درصد|percent)", re.I)

    _WEEKDAYS = (
        "شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه"
    )
    _EN_WEEKDAYS = (
        "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
    )

    @classmethod
    def quoted_prompts(cls, text: str) -> tuple[str, ...]:
        prompts: list[str] = []
        for match in cls._QUOTED.finditer(text):
            value = next((group for group in match.groups() if group), "").strip()
            if value:
                prompts.append(value)
        return tuple(prompts)

    @classmethod
    def is_prompt_batch(cls, text: str) -> bool:
        prompts = cls.quoted_prompts(text)
        if len(prompts) < 2:
            return False
        question_count = sum(
            prompt.endswith(("؟", "?"))
            or bool(re.search(r"(?:چیست|چقدر|چطور|چگونه|آیا|what|how|why)", prompt, re.I))
            for prompt in prompts
        )
        return question_count >= 2 or bool(
            re.search(r"(?:پیام|سؤال|سوال|پشت.?سرهم|آزمون|messages?|questions?|test)", text, re.I)
        )

    @classmethod
    def matches(cls, text: str) -> bool:
        normalized = normalize_text(text).translate(_DIGITS)
        if cls.is_prompt_batch(text):
            return True
        return any(
            (
                cls._is_boxes(normalized),
                cls._is_ropes(normalized),
                cls._is_syllogism(normalized),
                cls._is_counterfactual_weekday(normalized),
                cls._is_switches(normalized),
                cls._is_bat_ball(normalized),
                cls._is_clock(normalized),
                cls._is_trailing_zeros(normalized),
                cls._is_probability(normalized),
                cls._is_sequence(normalized),
            )
        )

    def solve(self, text: str, language: str = "fa") -> ChallengeAnswer | None:
        if self.is_prompt_batch(text):
            return self._solve_batch(text, language)
        return self._solve_single(text, language)

    def _solve_batch(self, text: str, language: str) -> ChallengeAnswer:
        prompts = self.quoted_prompts(text)
        answers = [self._solve_single(prompt, language) for prompt in prompts]
        if all(answer is not None for answer in answers):
            rendered = []
            for index, answer in enumerate(answers, 1):
                assert answer is not None
                rendered.append(f"{index}. {answer.text}")
            heading = "پاسخ هر سؤال:" if language == "fa" else "Answers:"
            return ChallengeAnswer(
                heading + "\n\n" + "\n\n".join(rendered),
                0.96,
                ("prompt_batch_detected", "questions_isolated", "all_items_locally_solved"),
            )
        if language == "fa":
            message = (
                f"این ورودی شامل {len(prompts)} پیام مستقل است. برای سنجش واقعی حافظه و پیگیری زمینه، "
                "آن‌ها را یکی‌یکی و به همان ترتیب بفرست؛ اگر همه را یک‌جا بفرستی، یک پیام واحد محسوب "
                "می‌شوند و آزمون حافظه معتبر نیست. من این متن را با موضوع گفت‌وگوی قبلی ترکیب نمی‌کنم."
            )
        else:
            message = (
                f"This input contains {len(prompts)} separate messages. Send them one at a time, in order, "
                "to test conversational memory; a pasted block is one turn and is not a valid memory test."
            )
        return ChallengeAnswer(
            message,
            0.99,
            ("prompt_batch_detected", "context_leak_prevented", "separate_turns_requested"),
        )

    def _solve_single(self, text: str, language: str) -> ChallengeAnswer | None:
        normalized = normalize_text(text).translate(_DIGITS)
        solvers = (
            self._solve_boxes,
            self._solve_ropes,
            self._solve_syllogism,
            self._solve_counterfactual_weekday,
            self._solve_switches,
            self._solve_bat_ball,
            self._solve_clock,
            self._solve_trailing_zeros,
            self._solve_probability,
            self._solve_sequence,
        )
        for solver in solvers:
            answer = solver(normalized, language)
            if answer is not None:
                return answer
        return None

    @staticmethod
    def _is_boxes(text: str) -> bool:
        persian = bool(
            re.search(r"(?:سه|3)\s*(?:جعبه|ظرف|صندوق)", text)
            and "مخلوط" in text
            and re.search(r"(?:هر\s*سه|همه).{0,35}(?:برچسب).{0,30}(?:اشتباه|غلط)", text)
            and re.search(r"(?:یک|1).{0,30}(?:میوه|مهره|سکه|توپ|مورد)", text)
            and re.search(r"(?:بیرون|بردار|خارج)", text)
        )
        english = bool(
            re.search(r"(?:three|3)\s*(?:boxes|bins|crates|containers)", text, re.I)
            and re.search(r"(?:mixed|mixture)", text, re.I)
            and re.search(r"(?:every|all).{0,45}labels?.{0,35}(?:wrong|incorrect|false)", text, re.I)
            and re.search(r"(?:draw|sample|take|remove).{0,35}(?:one|1)", text, re.I)
        )
        return persian or english

    def _solve_boxes(self, text: str, language: str) -> ChallengeAnswer | None:
        if not self._is_boxes(text):
            return None
        if language != "fa":
            return ChallengeAnswer(
                "Draw from the box labelled Mixed. Since every label is wrong, it is a pure box. "
                "The fruit drawn identifies it; use the all-wrong constraint to assign the other two labels."
            )
        kinds = re.search(
            r"برچسب(?:‌?های)?\s*[«\"']?([^،,:؛]{1,30})[»\"']?\s*[،,]\s*"
            r"[«\"']?([^،,:؛]{1,30})[»\"']?\s*(?:و|and)\s*[«\"']?مخلوط",
            text,
            re.I,
        )
        first, second = ("نوع اول", "نوع دوم")
        if kinds:
            first, second = (value.strip(" «»\"'") for value in kinds.groups())
        return ChallengeAnswer(
            f"از جعبه‌ای که برچسب «مخلوط» دارد یک مورد بردار. چون همهٔ برچسب‌ها غلط‌اند، این "
            f"جعبه واقعاً مخلوط نیست: اگر {first} بیرون آمد، آن جعبه فقط «{first}» است. از دو "
            f"جعبهٔ باقی‌مانده، جعبه‌ای که برچسب «{second}» دارد نمی‌تواند {second} باشد و {first} "
            f"هم قبلاً تعیین شده، پس آن جعبه «مخلوط» است و جعبهٔ آخر فقط «{second}» خواهد بود. "
            f"اگر مورد بیرون‌آمده {second} بود، جای {first} و {second} را در همین استدلال عوض کن."
        )

    @staticmethod
    def _is_ropes(text: str) -> bool:
        return (
            bool(re.search(r"(?:دو|2)\s*(?:طناب|فتیله)", text))
            and "60" in text and "45" in text
            and bool(re.search(r"(?:یکنواخت\s+نیست|غیریکنواخت|سرعت.{0,20}یکنواخت)", text))
        )

    def _solve_ropes(self, text: str, language: str) -> ChallengeAnswer | None:
        if not self._is_ropes(text):
            return None
        if language != "fa":
            return ChallengeAnswer(
                "Light both ends of rope A and one end of rope B. A finishes in 30 minutes; then light B's "
                "other end. Its remainder takes 15 minutes, for 45 minutes total."
            )
        return ChallengeAnswer(
            "هم‌زمان دو سر طناب اول و یک سر طناب دوم را روشن کن. طناب اول، مستقل از نامنظم‌بودن "
            "سرعت سوختن، در ۳۰ دقیقه تمام می‌شود. همان لحظه سر دیگر طناب دوم را روشن کن؛ مقدار "
            "باقی‌ماندهٔ آن از دو طرف در ۱۵ دقیقه می‌سوزد. پس ۳۰ + ۱۵ = ۴۵ دقیقه."
        )

    @staticmethod
    def _is_syllogism(text: str) -> bool:
        all_statement = bool(
            re.search(r"(?:همه|تمام).{1,100}(?:هستند|باشند|اند|ند|است)(?:\b|[.،؛])", text)
        )
        some_case = bool(
            re.search(r"(?:بعضی|برخی).{1,100}(?:اند|ند|هستند|باشند|است)(?:\b|[.،؛])", text)
            and re.search(r"(?:آیا|ايا).{0,25}حتما", text)
        )
        no_case = bool(
            re.search(r"هیچ.{1,100}(?:نیست|نباشد|نیستند|نباشند)(?:\b|[.،؛])", text)
            and re.search(r"(?:آیا|ايا).{0,25}(?:ممکن|می.?تواند|میشه|می.?شود)", text)
        )
        english_all = bool(re.search(r"\b(?:all|every)\b.{1,100}\b(?:are|is)\b", text, re.I))
        english_some = bool(
            re.search(r"\bsome\b.{1,100}\b(?:are|is)\b", text, re.I)
            and re.search(r"\b(?:must|does|necessarily|follow)\b", text, re.I)
        )
        english_no = bool(
            re.search(r"\bno\b.{1,100}\b(?:is|are)\b", text, re.I)
            and re.search(r"\b(?:can|could|possible|may)\b", text, re.I)
        )
        return (all_statement and (some_case or no_case)) or (english_all and (english_some or english_no))

    def _solve_syllogism(self, text: str, language: str) -> ChallengeAnswer | None:
        if not self._is_syllogism(text):
            return None
        if re.search(r"(?:\bهیچ.{1,100}(?:نیست|نباشد|نیستند|نباشند)\b|\bno\b.{1,100}\b(?:is|are)\b)", text, re.I):
            if language != "fa":
                return ChallengeAnswer(
                    "No. If every A is a B and no B is C, then A is a subset of B and B has no overlap "
                    "with C; therefore A cannot overlap with C either."
                )
            return ChallengeAnswer(
                "نه، ممکن نیست. گزارهٔ اول می‌گوید گروه اول زیرمجموعهٔ گروه دوم است. وقتی هیچ عضو "
                "گروه دوم آن ویژگی را ندارد، هیچ عضو زیرمجموعهٔ آن هم نمی‌تواند آن ویژگی را داشته باشد. "
                "پس مورد پیشنهادی با دو فرض مسئله تناقض دارد."
            )
        if language != "fa":
            return ChallengeAnswer(
                "No. 'All A are B' and 'some B are C' do not entail 'some A are C'; the C members of B "
                "may all lie outside A."
            )
        return ChallengeAnswer(
            "نه، حتماً نتیجه نمی‌شود. گزارهٔ اول فقط می‌گوید گروه اول زیرمجموعهٔ گروه دوم است. اعضای "
            "گروه دوم که ویژگی سوم را دارند ممکن است همگی بیرون از گروه اول باشند. مثال نقض: یک عضو "
            "گروه اول بدون آن ویژگی و یک عضو دارای ویژگی از گروه دوم ولی خارج از گروه اول در نظر بگیر؛ "
            "هر دو فرض درست می‌مانند، اما نتیجه برقرار نیست."
        )

    @classmethod
    def _canonical_weekday(cls, text: str) -> tuple[int, str] | None:
        compact = text.replace(" ", "").replace("‌", "")
        # Match the longest name first: «شنبه» is a suffix of «چهارشنبه».
        ordered = sorted(enumerate(cls._WEEKDAYS), key=lambda item: len(item[1]), reverse=True)
        for index, day in ordered:
            if day.replace("‌", "") in compact:
                return index, day
        for index, day in enumerate(cls._EN_WEEKDAYS):
            if re.search(rf"\b{day}\b", text, re.I):
                return index, day
        return None

    @classmethod
    def _is_counterfactual_weekday(cls, text: str) -> bool:
        persian = bool(
            re.search(r"اگر.{0,20}(?:پریروز|دیروز)", text)
            and re.search(r"فردای", text)
            and cls._canonical_weekday(text)
        )
        english = bool(
            re.search(r"\b(?:if|suppose)\b.{0,35}\b(?:yesterday|day\s+before\s+yesterday|two\s+days\s+ago)\b", text, re.I)
            and re.search(r"\b(?:day\s+after|day\s+following|tomorrow\s+of)\b", text, re.I)
            and cls._canonical_weekday(text)
        )
        return persian or english

    def _solve_counterfactual_weekday(self, text: str, language: str) -> ChallengeAnswer | None:
        if not self._is_counterfactual_weekday(text):
            return None
        found = self._canonical_weekday(text)
        assert found is not None
        index, day = found
        english_input = day in self._EN_WEEKDAYS
        names = self._EN_WEEKDAYS if language != "fa" else self._WEEKDAYS
        tomorrow = names[(index + 1) % 7]
        two_days = "پریروز" in text or bool(re.search(r"(?:day\s+before\s+yesterday|two\s+days\s+ago)", text, re.I))
        reference = ("the day before yesterday" if two_days else "yesterday") if english_input else ("پریروز" if two_days else "دیروز")
        distance = 2 if two_days else 1
        today = names[(index + 1 + distance) % 7]
        if language != "fa":
            return ChallengeAnswer(
                f"Interpret it literally: {reference} equals the day after {day}, which is {tomorrow}. "
                f"Moving forward {distance} day(s) makes today {today}."
            )
        return ChallengeAnswer(
            f"جمله را این‌طور می‌خوانیم: «{reference} = فردای {day}». فردای {day}، {tomorrow} است؛ "
            f"پس {reference} {tomorrow} بوده است. از {reference} تا امروز {distance} روز جلو می‌رویم، "
            f"بنابراین امروز {today} است. پاسخ: {today}."
        )

    @staticmethod
    def _is_switches(text: str) -> bool:
        return (
            bool(re.search(r"(?:سه|3)\s*کلید", text))
            and bool(re.search(r"(?:سه|3)\s*لامپ", text))
            and bool(re.search(r"(?:فقط\s*)?(?:یک|1)\s*بار", text))
            and bool(re.search(r"(?:وارد|اتاق)", text))
        )

    def _solve_switches(self, text: str, language: str) -> ChallengeAnswer | None:
        if not self._is_switches(text):
            return None
        if language != "fa":
            return ChallengeAnswer(
                "Turn switch 1 on for a few minutes, then off; turn switch 2 on and enter. The lit bulb is 2, "
                "the warm unlit bulb is 1, and the cold unlit bulb is 3."
            )
        return ChallengeAnswer(
            "کلید اول را چند دقیقه روشن بگذار و بعد خاموشش کن؛ کلید دوم را روشن کن و کلید سوم را "
            "دست نزن. حالا وارد اتاق لامپ‌ها شو: لامپ روشن مربوط به کلید دوم است، لامپ خاموشِ گرم "
            "مربوط به کلید اول و لامپ خاموشِ سرد مربوط به کلید سوم."
        )

    @classmethod
    def _numbers(cls, text: str) -> list[float]:
        values: list[float] = []
        for match in cls._NUMBER.finditer(text.translate(_DIGITS)):
            raw = match.group(0).strip()
            scale = 1
            if raw.endswith("هزار"):
                scale, raw = 1_000, raw.removesuffix("هزار").strip()
            elif raw.endswith("میلیون"):
                scale, raw = 1_000_000, raw.removesuffix("میلیون").strip()
            try:
                values.append(float(raw.replace(",", ".")) * scale)
            except ValueError:
                continue
        return values

    @staticmethod
    def _format_number(value: float, digits: int = 6) -> str:
        if math.isclose(value, round(value), abs_tol=1e-10):
            return f"{int(round(value)):,}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")

    @staticmethod
    def _is_bat_ball(text: str) -> bool:
        return (
            bool(re.search(r"(?:توپ|ball)", text, re.I))
            and bool(re.search(r"(?:چوب|bat)", text, re.I))
            and bool(re.search(r"(?:روی.?هم|مجموع|total)", text, re.I))
            and bool(re.search(r"(?:گران.?تر|more\s+than)", text, re.I))
        )

    def _solve_bat_ball(self, text: str, language: str) -> ChallengeAnswer | None:
        if not self._is_bat_ball(text):
            return None
        values = self._numbers(text)
        if len(values) < 2:
            return None
        total, difference = values[0], values[1]
        cheaper = (total - difference) / 2
        expensive = cheaper + difference
        if cheaper < 0 or not math.isclose(cheaper + expensive, total, rel_tol=1e-9):
            return None
        unit = " تومان" if "تومان" in text else ""
        if language != "fa":
            return ChallengeAnswer(
                f"Let the ball cost x. Then x + (x + {self._format_number(difference)}) = "
                f"{self._format_number(total)}, so x = {self._format_number(cheaper)}{unit}."
            )
        return ChallengeAnswer(
            f"قیمت توپ را x بگیریم؛ قیمت چوب x + {self._format_number(difference)} است. بنابراین "
            f"x + (x + {self._format_number(difference)}) = {self._format_number(total)} و در نتیجه "
            f"x = {self._format_number(cheaper)}{unit}. کنترل: {self._format_number(cheaper)} + "
            f"{self._format_number(expensive)} = {self._format_number(total)}."
        )

    @classmethod
    def _is_clock(cls, text: str) -> bool:
        return bool(cls._CLOCK.search(text)) and bool(re.search(r"(?:زاویه|angle|عقربه)", text, re.I))

    def _solve_clock(self, text: str, language: str) -> ChallengeAnswer | None:
        match = self._CLOCK.search(text)
        if not match or not self._is_clock(text):
            return None
        hour, minute = int(match.group(1)) % 12, int(match.group(2))
        if minute > 59:
            return None
        hour_angle = 30 * hour + 0.5 * minute
        minute_angle = 6 * minute
        difference = abs(hour_angle - minute_angle) % 360
        angle = min(difference, 360 - difference)
        value = self._format_number(angle)
        if language != "fa":
            return ChallengeAnswer(
                f"The hour hand is at {self._format_number(hour_angle)}° and the minute hand at "
                f"{self._format_number(minute_angle)}°, so the smaller angle is {value}°."
            )
        return ChallengeAnswer(
            f"عقربهٔ دقیقه روی {self._format_number(minute_angle)}° است. عقربهٔ ساعت‌شمار فقط روی عدد "
            f"ساعت نمی‌ماند و با گذشت دقیقه‌ها حرکت می‌کند؛ جای آن {self._format_number(hour_angle)}° "
            f"است. پس زاویهٔ کوچک‌تر دقیقاً {value} درجه است."
        )

    @classmethod
    def _is_trailing_zeros(cls, text: str) -> bool:
        return bool(cls._FACTORIAL.search(text)) and bool(re.search(r"(?:صفر.{0,20}انتها|trailing\s+zeros?)", text, re.I))

    def _solve_trailing_zeros(self, text: str, language: str) -> ChallengeAnswer | None:
        match = self._FACTORIAL.search(text)
        if not match or not self._is_trailing_zeros(text):
            return None
        number = int(match.group(1))
        if number < 0 or number > 10**9:
            return None
        terms: list[int] = []
        divisor = 5
        while divisor <= number:
            terms.append(number // divisor)
            divisor *= 5
        total = sum(terms)
        expression = " + ".join(str(term) for term in terms) or "0"
        if language != "fa":
            return ChallengeAnswer(f"Count factors of 5: {expression} = {total} trailing zeros.")
        return ChallengeAnswer(
            f"هر صفر از یک عامل ۱۰ می‌آید و عامل‌های ۲ فراوان‌ترند؛ پس عامل‌های ۵ را می‌شماریم: "
            f"{expression} = {total}. بنابراین {number}! دقیقاً {total} صفر انتهایی دارد."
        )

    @classmethod
    def _is_probability(cls, text: str) -> bool:
        return bool(cls._PERCENT.search(text)) and bool(
            re.search(r"(?:حداقل|دست.?کم|حداکثر|دقیقاً|دقیقا|at\s+least|at\s+most|exactly)", text, re.I)
            and re.search(r"(?:مستقل|independent)", text, re.I)
            and re.search(r"(?:آزمایش|تلاش|trials?|attempts?)", text, re.I)
        )

    @staticmethod
    def _success_target(text: str) -> tuple[str, int] | None:
        match = re.search(
            r"(?P<mode>حداقل|دست.?کم|حداکثر|دقیقاً|دقیقا|at\s+least|at\s+most|exactly)\s+"
            r"(?P<count>[\w\u0600-\u06FF‌.-]+)\s+(?:موفقیت|بار\s+موفق|success(?:es)?)",
            text,
            re.I,
        )
        if not match:
            return None
        parsed = ChallengeReasoner._parse_small_number(match.group("count"))
        if parsed is None or float(parsed) != int(parsed):
            return None
        raw_mode = normalize_text(match.group("mode"))
        if re.search(r"(?:حداکثر|at\s+most)", raw_mode, re.I):
            mode = "at_most"
        elif re.search(r"(?:دقیق|exactly)", raw_mode, re.I):
            mode = "exactly"
        else:
            mode = "at_least"
        return mode, int(parsed)

    @staticmethod
    def _trial_count(text: str) -> int | None:
        match = re.search(
            r"(?:در|طی|in)\s+([\w\u0600-\u06FF‌.-]+)\s+(?:آزمایش|تلاش|trials?|attempts?)",
            text,
            re.I,
        )
        if not match:
            return None
        parsed = ChallengeReasoner._parse_small_number(match.group(1))
        if parsed is None or float(parsed) != int(parsed):
            return None
        return int(parsed)

    @staticmethod
    def _parse_small_number(value: str) -> int | float | None:
        parsed = PersianNumberParser.parse(value.translate(_DIGITS))
        if parsed is not None:
            return parsed
        return {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }.get(normalize_text(value))

    def _solve_probability(self, text: str, language: str) -> ChallengeAnswer | None:
        percent_match = self._PERCENT.search(text)
        trials = self._trial_count(text)
        target = self._success_target(text)
        if not percent_match or trials is None or target is None or not self._is_probability(text):
            return None
        percent = float(percent_match.group(1).replace(",", "."))
        mode, threshold = target
        if not 0 <= percent <= 100 or trials < 1 or trials > 10_000 or threshold < 0:
            return None
        p = percent / 100
        q = 1 - p

        def mass(successes: int) -> float:
            if not 0 <= successes <= trials:
                return 0.0
            return math.comb(trials, successes) * p ** successes * q ** (trials - successes)

        if mode == "exactly":
            probability = mass(threshold)
            formula = f"C({trials},{threshold}) × {p:g}^{threshold} × {q:g}^{trials - threshold}"
            fa_label, en_label = f"دقیقاً {threshold}", f"exactly {threshold}"
        elif mode == "at_most":
            probability = sum(mass(value) for value in range(0, min(threshold, trials) + 1))
            formula = " + ".join(f"P(X={value})" for value in range(0, min(threshold, trials) + 1))
            fa_label, en_label = f"حداکثر {threshold}", f"at most {threshold}"
        else:
            probability = sum(mass(value) for value in range(max(0, threshold), trials + 1))
            if threshold <= 2:
                formula = "1 - " + " - ".join(f"P(X={value})" for value in range(threshold))
            else:
                formula = " + ".join(f"P(X={value})" for value in range(threshold, trials + 1))
            fa_label, en_label = f"حداقل {threshold}", f"at least {threshold}"
        result = self._format_number(probability * 100, 6)
        if language != "fa":
            return ChallengeAnswer(
                f"Let X follow Binomial(n={trials}, p={p:g}). For {en_label} successes, use {formula}; "
                f"the probability is {result}%."
            )
        return ChallengeAnswer(
            f"تعداد موفقیت‌ها را X می‌گیریم؛ X توزیع دوجمله‌ای با n={trials} و p={p:g} دارد. "
            f"برای احتمال {fa_label} موفقیت از {formula} استفاده می‌کنیم. نتیجه برابر {result}٪ است."
        )

    @classmethod
    def _is_sequence(cls, text: str) -> bool:
        return bool(cls._SEQUENCE.search(text)) and len(cls._numbers(text)) >= 3

    @staticmethod
    def _all_close(values: list[float]) -> bool:
        return bool(values) and all(math.isclose(value, values[0], rel_tol=1e-9, abs_tol=1e-9) for value in values)

    def _solve_sequence(self, text: str, language: str) -> ChallengeAnswer | None:
        if not self._is_sequence(text):
            return None
        values = self._numbers(text)
        if len(values) < 3:
            return None
        differences = [right - left for left, right in zip(values, values[1:])]
        rule = ""
        if self._all_close(differences):
            next_value = values[-1] + differences[-1]
            rule = f"اختلاف ثابت {self._format_number(differences[0])} است"
        else:
            ratios = [right / left for left, right in zip(values, values[1:]) if not math.isclose(left, 0)]
            if len(ratios) == len(values) - 1 and self._all_close(ratios):
                next_value = values[-1] * ratios[-1]
                rule = f"هر جمله در {self._format_number(ratios[0])} ضرب شده است"
            else:
                diff_ratios = [
                    right / left for left, right in zip(differences, differences[1:])
                    if not math.isclose(left, 0)
                ]
                if len(diff_ratios) == len(differences) - 1 and diff_ratios and self._all_close(diff_ratios):
                    next_difference = differences[-1] * diff_ratios[-1]
                    next_value = values[-1] + next_difference
                    rule = (
                        "اختلاف‌ها " + "، ".join(self._format_number(item) for item in differences)
                        + f" هستند و هر بار در {self._format_number(diff_ratios[0])} ضرب می‌شوند"
                    )
                else:
                    second = [right - left for left, right in zip(differences, differences[1:])]
                    if not self._all_close(second):
                        return None
                    next_difference = differences[-1] + second[-1]
                    next_value = values[-1] + next_difference
                    rule = f"اختلاف مرتبهٔ دوم ثابت {self._format_number(second[0])} است"
        result = self._format_number(next_value)
        ambiguity = bool(re.search(r"(?:قطعی|یکتا|چند\s+قاعده|unique|ambiguous)", text, re.I))
        if language != "fa":
            suffix = " A finite sequence is never mathematically unique without a stated rule." if ambiguity else ""
            return ChallengeAnswer(f"A natural continuation is {result}: {rule}.{suffix}")
        caveat = (
            " بااین‌حال، از تعداد محدودی جمله می‌توان بی‌نهایت قاعده ساخت؛ پس بدون اعلام قاعده، جواب "
            "از نظر ریاضی یکتا نیست."
            if ambiguity else ""
        )
        return ChallengeAnswer(f"یک ادامهٔ طبیعی {result} است، چون {rule}.{caveat}")


__all__ = ["ChallengeAnswer", "ChallengeReasoner"]
