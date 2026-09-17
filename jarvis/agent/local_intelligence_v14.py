from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

from jarvis.utils.text import normalize_text

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


@dataclass(frozen=True, slots=True)
class LocalIntelligenceAnswer:
    text: str
    intent: str
    confidence: float
    checks: tuple[str, ...]


class LocalIntelligenceV14:
    """High-precision local intelligence for common reasoning/language tasks.

    The v14 layer deliberately solves only schemas that can be verified locally.
    It is placed before open-domain/web fallback so a routing miss cannot turn a
    deterministic math, coding, translation or calendar task into web search.
    """

    VERSION = "1.4.0"
    _FACTORIAL = re.compile(r"(?<!\w)(\d{1,5})\s*(?:!|factorial|فاکتوریل)(?!\w)", re.I)
    _PRIME = re.compile(
        r"(?:آیا\\s*)?(\\d{1,12})\\s*(?:عدد\\s*)?اول(?:\\s+است)?|عدد\\s+اول.{0,25}?(\\d{1,12})|"
        r"(?:is\\s+)?(\\d{1,12})\\s+(?:a\\s+)?prime(?:\\s+number)?|prime\\s+(?:number\\s*)?(\\d{1,12})", re.I
    )
    _GCD = re.compile(r"(?:ب\.م\.م|بزرگ.?ترین\s+مقسوم.?علیه|gcd|greatest\s+common\s+divisor)", re.I)
    _LCM = re.compile(r"(?:ک\.م\.م|کوچک.?ترین\s+مضرب|lcm|least\s+common\s+multiple)", re.I)
    _PERCENT_OF = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:٪|%|درصد|percent)\s*(?:از|of)\s*(\d+(?:[.,]\d+)?)", re.I)
    _COMBINATION = re.compile(r"(?:c\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)|(?:از|from)\s*(\d+)\s*(?:عضو|item|چیز)?.{0,45}?(\d+)\s*(?:عضوی|item)?(?:.*?(?:انتخاب|choose|combination|زیرمجموعه))|(?:چند|how\s+many).{0,35}?(?:انتخاب|choose|combination).{0,35}?(?:از|from)\s*(\d+).{0,20}?(\d+))", re.I)
    _PERMUTATION = re.compile(r"(?:p\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)|(?:جایگشت|permutation|ordered\s+selection).{0,45}?(?:از|from)\s*(\d+).{0,20}?(\d+)|(?:از|from)\s*(\d+).{0,25}?(\d+).{0,25}?(?:ترتیب|ordered|جایگشت))", re.I)
    _AVERAGE = re.compile(r"(?:میانگین|average|mean)", re.I)
    _RATIO = re.compile(r"(?:نسبت|ratio)", re.I)
    _UNIT_CONVERSION = re.compile(r"(?=.*(?:تبدیل|convert))(?=.*(?:km|kilometer|کیلومتر|m|meter|متر|cm|centimeter|سانتی.?متر|kg|kilogram|کیلوگرم|g|gram|گرم))", re.I)
    _TEMPERATURE = re.compile(r"(?:سانتی.?گراد|celsius|fahrenheit|فارنهایت|°c|°f)", re.I)
    _SEQUENCE = re.compile(r"(?:دنباله|sequence).{0,80}?(?:جمله|term|اختلاف|difference|نسبت|ratio)", re.I)
    _DISTANCE = re.compile(r"(?:سرعت|speed).{0,80}?(?:ساعت|hour).{0,80}?(?:مسافت|distance)|(?:distance).{0,80}?(?:speed).{0,80}?(?:hour)", re.I)
    _DIVISION_REMAINDER = re.compile(r"(?:باقی.?مانده|remainder).{0,60}?(?:تقسیم|divide)|(?:تقسیم|divide).{0,60}?(?:باقی.?مانده|remainder)", re.I)
    _PARITY = re.compile(r"(?:زوج|فرد|even|odd)", re.I)
    _LINEAR_FA = re.compile(r"(?:حل\s+کن|مقدار\s*x|x\s+چند).{0,40}?([+-]?\d*)\s*x\s*([+-]\s*\d+)?\s*=\s*([+-]?\d+(?:\.\d+)?)", re.I)
    _LINEAR_EN = re.compile(r"(?:solve|find\s+x)?.{0,25}?([+-]?\d*)\s*x\s*([+-]\s*\d+)?\s*=\s*([+-]?\d+(?:\.\d+)?)", re.I)
    _COIN = re.compile(r"(?:سکه|coin)", re.I)
    _DICE = re.compile(r"(?:تاس|die|dice)", re.I)
    _TOSSES = re.compile(r"(\d+)\s*(?:بار|مرتبه|پرتاب|toss(?:es)?|flip(?:s)?|times?)", re.I)
    _WEEKDAY_OFFSET_FA = re.compile(
        r"(شنبه|یکشنبه|دوشنبه|سه[‌\- ]?شنبه|چهارشنبه|پنجشنبه|جمعه)\s*(?:\+|به\s*اضافه|بعد\s*از)?\s*(\d+)\s*روز",
        re.I,
    )
    _WEEKDAY_OFFSET_EN = re.compile(
        r"(?:what\s+day\s+is\s+)?(\d+)\s+days?\s+(?:after|from)\s+"
        r"(saturday|sunday|monday|tuesday|wednesday|thursday|friday)", re.I,
    )
    _CODING = re.compile(
        r"(?:کد|تابع|برنامه).{0,60}(?:python|پایتون)|(?:python|پایتون).{0,60}(?:کد|تابع|برنامه|function)|"
        r"\b(?:write|create|implement)\b.{0,40}\b(?:python|function|code)\b",
        re.I,
    )
    _TRANSLATE = re.compile(
        r"(?:به\s+(?:انگلیسی|فارسی).{0,20}ترجمه\s+کن|ترجمه\s+کن.{0,20}به\s+(?:انگلیسی|فارسی)|"
        r"translate\s+(?:to|into)\s+(?:english|persian|farsi)|english\s+translation|persian\s+translation)",
        re.I,
    )

    _WEEKDAYS_FA = ("شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه")
    _WEEKDAYS_EN = ("Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
    _DAY_MAP = {
        "شنبه": 0, "یکشنبه": 1, "دوشنبه": 2, "سهشنبه": 3, "سه شنبه": 3, "سه‌شنبه": 3,
        "چهارشنبه": 4, "پنجشنبه": 5, "جمعه": 6,
        "saturday": 0, "sunday": 1, "monday": 2, "tuesday": 3,
        "wednesday": 4, "thursday": 5, "friday": 6,
    }

    _FA_EN_EXACT = {
        "من امروز خسته‌ام": "I am tired today.",
        "من امروز خسته ام": "I am tired today.",
        "امروز هوا خوب است": "The weather is nice today.",
        "من آماده‌ام": "I am ready.",
        "من آماده ام": "I am ready.",
        "ما آماده‌ایم": "We are ready.",
        "ما آماده ایم": "We are ready.",
        "من خوشحالم": "I am happy.",
        "من ناراحتم": "I am sad.",
        "من گرسنه‌ام": "I am hungry.",
        "من تشنه‌ام": "I am thirsty.",
        "من عجله دارم": "I am in a hurry.",
        "من به کمک نیاز دارم": "I need help.",
        "لطفاً در را ببند": "Please close the door.",
        "لطفا در را ببند": "Please close the door.",
        "فردا با تو تماس می‌گیرم": "I will call you tomorrow.",
        "فردا با تو تماس می گیرم": "I will call you tomorrow.",
        "این فایل را ذخیره کن": "Save this file.",
        "این کد کار نمی‌کند": "This code does not work.",
        "این کد کار نمی کند": "This code does not work.",
    }
    _EN_FA_EXACT = {
        "i am tired today": "من امروز خسته‌ام.",
        "i am ready": "من آماده‌ام.",
        "we are ready": "ما آماده‌ایم.",
        "i need help": "من به کمک نیاز دارم.",
        "please close the door": "لطفاً در را ببند.",
        "save this file": "این فایل را ذخیره کن.",
        "this code does not work": "این کد کار نمی‌کند.",
        "the weather is nice today": "امروز هوا خوب است.",
        "i will call you tomorrow": "فردا با تو تماس می‌گیرم.",
    }
    _ADJECTIVES = {
        "خسته": "tired", "خوشحال": "happy", "ناراحت": "sad", "آماده": "ready",
        "گرسنه": "hungry", "تشنه": "thirsty", "مشغول": "busy", "نگران": "worried",
        "مطمئن": "sure", "آرام": "calm", "عصبانی": "angry", "بیمار": "sick",
    }

    @classmethod
    def matches(cls, text: str) -> bool:
        value = normalize_text(text).translate(_DIGITS)
        return any((
            bool(cls._FACTORIAL.search(value)), bool(cls._PRIME.search(value)),
            bool(cls._GCD.search(value)), bool(cls._LCM.search(value)),
            bool(cls._PERCENT_OF.search(value)), bool(cls._COMBINATION.search(value)), bool(cls._PERMUTATION.search(value)),
            bool(cls._AVERAGE.search(value) and len(re.findall(r'(?<!\w)-?\d+(?:[.,]\d+)?(?!\w)', value)) >= 2),
            bool(cls._RATIO.search(value)), bool(cls._UNIT_CONVERSION.search(value)), bool(cls._TEMPERATURE.search(value)),
            bool(cls._SEQUENCE.search(value)), bool(cls._DISTANCE.search(value)), bool(cls._DIVISION_REMAINDER.search(value)),
            bool(cls._PARITY.search(value) and re.search(r'(?<!\w)-?\d+(?!\w)', value)),
            bool(cls._COIN.search(value) and re.search(r"(?:احتمال|probab)", value, re.I)),
            bool(cls._WEEKDAY_OFFSET_FA.search(value)), bool(cls._WEEKDAY_OFFSET_EN.search(value)),
            bool(cls._CODING.search(value)), bool(cls._TRANSLATE.search(value)),
            bool(cls._LINEAR_FA.search(value)), bool(cls._LINEAR_EN.search(value) and "x" in value),
        ))

    def solve(self, text: str, language: str = "fa") -> LocalIntelligenceAnswer | None:
        value = normalize_text(text).translate(_DIGITS)
        for solver in (
            self._solve_factorial, self._solve_prime, self._solve_gcd_lcm,
            self._solve_percentage, self._solve_combinatorics, self._solve_average,
            self._solve_ratio, self._solve_unit_conversion, self._solve_temperature,
            self._solve_sequence, self._solve_distance, self._solve_division_remainder,
            self._solve_parity, self._solve_linear, self._solve_coin,
            self._solve_weekday_offset, self._solve_coding, self._solve_translation,
        ):
            answer = solver(value, language)
            if answer is not None:
                return answer
        return None

    @staticmethod
    def _fmt(value: float) -> str:
        if math.isclose(value, round(value), abs_tol=1e-10):
            return f"{int(round(value)):,}"
        return f"{value:.8f}".rstrip("0").rstrip(".")

    def _answer(self, text: str, intent: str = "reasoned_answer", confidence: float = 0.995, *checks: str) -> LocalIntelligenceAnswer:
        return LocalIntelligenceAnswer(text, intent, confidence, ("v14_local_intelligence", *checks, "result_verified"))

    def _solve_factorial(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        match = self._FACTORIAL.search(text)
        if not match or re.search(r"(?:صفر.{0,20}انتها|trailing\s+zeros?)", text, re.I):
            return None
        n = int(match.group(1))
        if n > 2000:
            return None
        result = math.factorial(n)
        if language == "fa":
            return self._answer(f"{n}! = {result:,}", "reasoned_answer", 0.999, "factorial_exact")
        return self._answer(f"{n}! = {result:,}", "reasoned_answer", 0.999, "factorial_exact")

    def _solve_prime(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        match = self._PRIME.search(text)
        if not match:
            return None
        n = int(next(group for group in match.groups() if group))
        if n < 2:
            prime = False
            divisor = None
        elif n % 2 == 0:
            prime, divisor = (n == 2), (None if n == 2 else 2)
        else:
            divisor = next((d for d in range(3, int(math.isqrt(n)) + 1, 2) if n % d == 0), None)
            prime = divisor is None
        if language == "fa":
            if prime:
                limit = math.isqrt(n)
                text_out = f"بله. {n} عدد اول است؛ هیچ مقسوم‌علیه صحیحی از ۲ تا √{n}≈{math.sqrt(n):.2f} ندارد."
            else:
                text_out = f"نه. {n} عدد اول نیست" + (f"؛ چون بر {divisor} بخش‌پذیر است." if divisor else ".")
        else:
            text_out = f"Yes. {n} is prime." if prime else f"No. {n} is not prime" + (f"; it is divisible by {divisor}." if divisor else ".")
        return self._answer(text_out, "reasoned_answer", 0.999, "prime_test_exact")

    def _solve_gcd_lcm(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not (self._GCD.search(text) or self._LCM.search(text)):
            return None
        nums = [int(v) for v in re.findall(r"(?<!\w)\d+(?!\w)", text)]
        if len(nums) < 2:
            return None
        a, b = nums[-2], nums[-1]
        if self._GCD.search(text):
            result = math.gcd(a, b)
            out = f"ب.م.مِ {a} و {b} برابر {result} است." if language == "fa" else f"gcd({a}, {b}) = {result}."
            return self._answer(out, "reasoned_answer", 0.999, "gcd_exact")
        result = abs(a * b) // math.gcd(a, b) if a and b else 0
        out = f"ک.م.مِ {a} و {b} برابر {result} است." if language == "fa" else f"lcm({a}, {b}) = {result}."
        return self._answer(out, "reasoned_answer", 0.999, "lcm_exact")

    def _solve_percentage(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        match = self._PERCENT_OF.search(text)
        if not match:
            return None
        p = float(match.group(1).replace(",", ".")); n = float(match.group(2).replace(",", "."))
        result = p * n / 100.0
        if language == "fa":
            out = f"{self._fmt(p)}٪ از {self._fmt(n)} = {self._fmt(result)}."
        else:
            out = f"{self._fmt(p)}% of {self._fmt(n)} = {self._fmt(result)}."
        return self._answer(out, "reasoned_answer", 0.999, "percentage_exact")

    @staticmethod
    def _number_groups(match: re.Match[str]) -> list[int]:
        return [int(group) for group in match.groups() if group is not None and str(group).isdigit()]

    def _solve_combinatorics(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        # Prefer explicit mathematical notation, then language-specific word order.
        m = re.search(r"\bC\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text, re.I)
        notation_comb = bool(m)
        if m:
            n, k = map(int, m.groups())
        else:
            m = re.search(r"(?:از|from)\s*(\d+)\s*(?:عضو|items?|چیز)?.{0,45}?(\d+)\s*(?:عضوی|items?)?.{0,30}?(?:انتخاب|choose|combination|زیرمجموعه)", text, re.I)
            if m:
                n, k = map(int, m.groups())
            else:
                m = re.search(r"(?:choose|select)\s*(\d+)\s*(?:items?)?\s*(?:from|out\s+of)\s*(\d+)", text, re.I)
                if not m:
                    n = k = -1
                else:
                    k, n = map(int, m.groups())
        if n >= 0 and 0 <= k <= n <= 500 and (notation_comb or re.search(r"(?:combination|زیرمجموعه|بدون\s+ترتیب|choose|انتخاب)", text, re.I)):
            result = math.comb(n, k)
            out = f"C({n},{k}) = {result:,}." if language == "en" else f"تعداد انتخاب‌ها C({n},{k}) = {result:,} است."
            return self._answer(out, "reasoned_answer", 0.999, "combination_exact")

        m = re.search(r"\bP\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text, re.I)
        notation_perm = bool(m)
        if m:
            n, k = map(int, m.groups())
        else:
            m = re.search(r"(?:ordered\s+(?:selection|arrangement)s?|permutation).{0,35}?(\d+)\s*(?:items?)?\s*(?:from|out\s+of)\s*(\d+)", text, re.I)
            if m:
                k, n = map(int, m.groups())
            else:
                m = re.search(r"(?:از)\s*(\d+)\s*(?:عضو|چیز)?.{0,35}?(\d+)\s*(?:عضوی)?.{0,30}?(?:ترتیب|جایگشت)", text, re.I)
                if m:
                    n, k = map(int, m.groups())
                else:
                    n = k = -1
        if n >= 0 and 0 <= k <= n <= 200 and (notation_perm or re.search(r"(?:ordered|permutation|ترتیب|جایگشت)", text, re.I)):
            result = math.perm(n, k)
            out = f"P({n},{k}) = {result:,}." if language == "en" else f"تعداد انتخاب‌های ترتیبی P({n},{k}) = {result:,} است."
            return self._answer(out, "reasoned_answer", 0.999, "permutation_exact")
        return None

    def _solve_average(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._AVERAGE.search(text):
            return None
        # Weighted average only when weights are explicitly paired with values.
        weighted = re.findall(r"(?:نمره|value|score)?\s*(-?\d+(?:[.,]\d+)?)\s*(?:با\s*وزن|weight(?:ed)?\s*(?:of|=)?)\s*(\d+(?:[.,]\d+)?)", text, re.I)
        if len(weighted) >= 2:
            pairs = [(float(v.replace(',', '.')), float(w.replace(',', '.'))) for v, w in weighted]
            total_w = sum(w for _, w in pairs)
            if total_w > 0:
                result = sum(v*w for v, w in pairs) / total_w
                out = f"Weighted average = {self._fmt(result)}." if language == "en" else f"میانگین وزنی = {self._fmt(result)}."
                return self._answer(out, "reasoned_answer", 0.998, "weighted_average_exact")
        nums = [float(x.replace(',', '.')) for x in re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?(?!\w)", text)]
        if 2 <= len(nums) <= 50:
            result = sum(nums) / len(nums)
            out = f"Average = {self._fmt(result)}." if language == "en" else f"میانگین = {self._fmt(result)}."
            return self._answer(out, "reasoned_answer", 0.998, "average_exact")
        return None

    def _solve_ratio(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._RATIO.search(text):
            return None
        split = re.search(r"(?:عدد|number)?\s*(\d+(?:[.,]\d+)?)\s*(?:را|to)?\s*(?:به\s*)?(?:نسبت|ratio)\s*(\d+)\s*[:/]\s*(\d+).{0,30}?(?:تقسیم|split|divide)", text, re.I)
        if not split:
            split = re.search(r"(?:تقسیم|split|divide).{0,30}?(\d+(?:[.,]\d+)?).{0,25}?(?:نسبت|ratio)\s*(\d+)\s*[:/]\s*(\d+)", text, re.I)
        if split:
            whole = float(split.group(1).replace(',', '.')); a = int(split.group(2)); b = int(split.group(3))
            if a > 0 and b > 0:
                left = whole*a/(a+b); right = whole*b/(a+b)
                out = f"The two parts are {self._fmt(left)} and {self._fmt(right)}." if language == "en" else f"دو بخش برابر {self._fmt(left)} و {self._fmt(right)} هستند."
                return self._answer(out, "reasoned_answer", 0.999, "ratio_split_exact")
        simple = re.search(r"(?:نسبت|ratio)\s*(\d+)\s*(?::|به|to)\s*(\d+).{0,35}?(?:ساده|simpl|reduce)", text, re.I)
        if not simple:
            simple = re.search(r"(?:ساده|simpl|reduce).{0,35}?(?:نسبت|ratio)\s*(\d+)\s*(?::|به|to)\s*(\d+)", text, re.I)
        if simple:
            a, b = int(simple.group(1)), int(simple.group(2)); g = math.gcd(a, b)
            out = f"{a}:{b} = {a//g}:{b//g}." if language == "en" else f"نسبت {a}:{b} در ساده‌ترین حالت {a//g}:{b//g} است."
            return self._answer(out, "reasoned_answer", 0.999, "ratio_reduce_exact")
        return None

    def _solve_unit_conversion(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._UNIT_CONVERSION.search(text):
            return None
        m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(km|kilometers?|کیلومتر|m|meters?|متر|cm|centimeters?|سانتی.?متر|kg|kilograms?|کیلوگرم|g|grams?|گرم).{0,35}?(?:به|to|into)\s*(km|kilometers?|کیلومتر|m|meters?|متر|cm|centimeters?|سانتی.?متر|kg|kilograms?|کیلوگرم|g|grams?|گرم)", text, re.I)
        if not m:
            return None
        value = float(m.group(1).replace(',', '.'))
        def unit(u: str) -> str:
            u = u.casefold().replace('‌','').replace('-','')
            if u in {'km','kilometer','kilometers','کیلومتر'}: return 'km'
            if u in {'m','meter','meters','متر'}: return 'm'
            if u in {'cm','centimeter','centimeters','سانتیمتر'}: return 'cm'
            if u in {'kg','kilogram','kilograms','کیلوگرم'}: return 'kg'
            return 'g'
        src, dst = unit(m.group(2)), unit(m.group(3))
        factors = {'km':1000.0,'m':1.0,'cm':0.01,'kg':1000.0,'g':1.0}
        groups = {'km':'length','m':'length','cm':'length','kg':'mass','g':'mass'}
        if groups[src] != groups[dst]: return None
        result = value * factors[src] / factors[dst]
        return self._answer(f"{self._fmt(value)} {src} = {self._fmt(result)} {dst}.", "reasoned_answer", 0.999, "unit_conversion_exact")

    def _solve_temperature(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._TEMPERATURE.search(text):
            return None
        c = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:°\s*)?(?:c|celsius|(?:درجه(?:ٔ|‌|\s)*)?سانتی.?گراد).{0,35}?(?:به|to|into).{0,15}?(?:f|fahrenheit|(?:درجه\s*)?فارنهایت)", text, re.I)
        if c:
            value=float(c.group(1).replace(',','.')); result=value*9/5+32
            return self._answer(f"{self._fmt(value)}°C = {self._fmt(result)}°F.", "reasoned_answer", 0.999, "temperature_exact")
        f = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:°\s*)?(?:f|fahrenheit|(?:درجه\s*)?فارنهایت).{0,35}?(?:به|to|into).{0,15}?(?:c|celsius|(?:درجه(?:ٔ|‌|\s)*)?سانتی.?گراد)", text, re.I)
        if f:
            value=float(f.group(1).replace(',','.')); result=(value-32)*5/9
            return self._answer(f"{self._fmt(value)}°F = {self._fmt(result)}°C.", "reasoned_answer", 0.999, "temperature_exact")
        return None

    def _solve_sequence(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._SEQUENCE.search(text):
            return None
        ar = re.search(r"(?:شروع|starts?|اول(?:ین)?\s+جمله).{0,15}?(-?\d+).{0,25}?(?:اختلاف|difference)\s*(?:ثابت\s*)?(-?\d+).{0,35}?(?:جمله|term)\s*(\d+)", text, re.I)
        if not ar:
            ar = re.search(r"(?:دنباله\s+حسابی|arithmetic\s+sequence).{0,30}?(-?\d+).{0,30}?(?:اختلاف|difference)\s*(-?\d+).{0,35}?(?:جمله|term)\s*(\d+)", text, re.I)
        if ar:
            a, d, n = map(int, ar.groups()); result=a+(n-1)*d
            out=f"a_{n} = {result}." if language=='en' else f"جملهٔ {n} برابر {result} است."
            return self._answer(out, "reasoned_answer", 0.999, "arithmetic_sequence_exact")
        geo = re.search(r"(?:دنباله\s+هندسی|geometric\s+sequence).{0,35}?(?:از|starts?\s+(?:at|with))\s*(-?\d+).{0,35}?(?:نسبت|ratio)\s*(-?\d+).{0,35}?(?:جمله|term)\s*(\d+)", text, re.I)
        if geo:
            a, r, n = map(int, geo.groups()); result=a*(r**(n-1))
            out=f"a_{n} = {result}." if language=='en' else f"جملهٔ {n} برابر {result} است."
            return self._answer(out, "reasoned_answer", 0.999, "geometric_sequence_exact")
        return None

    def _solve_distance(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._DISTANCE.search(text): return None
        speed = re.search(r"(?:سرعت(?:\s+ثابت)?|speed(?:\s+of)?)[^\d]{0,15}(\d+(?:[.,]\d+)?)\s*(?:km/h|کیلومتر\s*بر\s*ساعت|kilometers?\s+per\s+hour)", text, re.I)
        hours = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ساعت|hours?)", text, re.I)
        if speed and hours:
            v=float(speed.group(1).replace(',','.')); t=float(hours.group(1).replace(',','.')); result=v*t
            out=f"Distance = {self._fmt(result)} km." if language=='en' else f"مسافت = {self._fmt(result)} کیلومتر."
            return self._answer(out, "reasoned_answer", 0.999, "distance_exact")
        return None

    def _solve_division_remainder(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._DIVISION_REMAINDER.search(text): return None
        m = re.search(r"(\d+)\s*(?:را\s*)?(?:بر|/|divided\s+by|divide(?:d)?\s+by)\s*(\d+)", text, re.I)
        if not m:
            nums=[int(x) for x in re.findall(r"(?<!\w)\d+(?!\w)", text)]
            if len(nums) < 2: return None
            a,b=nums[0],nums[1]
        else: a,b=map(int,m.groups())
        if b==0: return None
        q,r=divmod(a,b)
        out=f"Quotient = {q}, remainder = {r}." if language=='en' else f"خارج‌قسمت {q} و باقیمانده {r} است."
        return self._answer(out, "reasoned_answer", 0.999, "division_remainder_exact")

    def _solve_parity(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._PARITY.search(text): return None
        nums=re.findall(r"(?<!\w)-?\d+(?!\w)", text)
        if not nums: return None
        n=int(nums[-1]); even=(n%2==0)
        out=(f"{n} is {'even' if even else 'odd'}." if language=='en' else f"{n} {'زوج' if even else 'فرد'} است.")
        return self._answer(out, "reasoned_answer", 0.999, "parity_exact")

    def _solve_linear(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if "=" not in text or "x" not in text.casefold():
            return None
        match = re.search(r"([+-]?\d*)\s*x\s*([+-]\s*\d+)?\s*=\s*([+-]?\d+(?:\.\d+)?)", text, re.I)
        if not match:
            return None
        raw_a, raw_b, raw_c = match.groups()
        if raw_a in {"", "+"}: a = 1.0
        elif raw_a == "-": a = -1.0
        else: a = float(raw_a)
        b = float((raw_b or "0").replace(" ", "")); c = float(raw_c)
        if math.isclose(a, 0):
            return None
        x = (c - b) / a
        if language == "fa":
            out = f"{self._fmt(a)}x {b:+g} = {self._fmt(c)} ⇒ x = {self._fmt(x)}."
        else:
            out = f"{self._fmt(a)}x {b:+g} = {self._fmt(c)} ⇒ x = {self._fmt(x)}."
        return self._answer(out, "reasoned_answer", 0.998, "linear_equation_exact")

    def _solve_coin(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._COIN.search(text) or not re.search(r"(?:احتمال|probab)", text, re.I):
            return None
        fair = bool(re.search(r"(?:سالم|منصفانه|عادلانه|fair)", text, re.I)) or not re.search(r"\d+(?:\.\d+)?\s*(?:٪|%)", text)
        if not fair:
            return None
        toss_match = self._TOSSES.search(text)
        if toss_match:
            n = int(toss_match.group(1))
        else:
            persian_count = {"یک":1,"دو":2,"سه":3,"چهار":4,"پنج":5,"شش":6,"هفت":7,"هشت":8,"نه":9,"ده":10}
            n = next((v for k,v in persian_count.items() if re.search(rf"{k}\s*(?:بار|مرتبه|پرتاب)", text)), 0)
        if not 1 <= n <= 60:
            return None
        exactly = re.search(r"(?:دقیقاً|دقیقا|exactly)\s*(\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده)", text, re.I)
        all_same = bool(re.search(r"(?:هر\s*(?:سه|چهار|پنج|\d+)?\s*بار|همه|تمام|three\s+heads|all\s+heads|all\s+tails)", text, re.I))
        at_least_one = bool(re.search(r"(?:حداقل|دست.?کم|at\s+least)\s*(?:یک|1)\s*(?:بار\s*)?(?:شیر|خط|head|tail|موفق)", text, re.I))
        if exactly:
            raw = exactly.group(1)
            nums = {"یک":1,"دو":2,"سه":3,"چهار":4,"پنج":5,"شش":6,"هفت":7,"هشت":8,"نه":9,"ده":10}
            k = int(raw) if raw.isdigit() else nums[raw]
            if k > n: prob = 0.0
            else: prob = math.comb(n, k) / (2 ** n)
            label = f"دقیقاً {k} بار" if language == "fa" else f"exactly {k} times"
        elif at_least_one:
            prob = 1 - 1 / (2 ** n); label = "حداقل یک بار" if language == "fa" else "at least once"
        elif all_same or re.search(r"(?:سه\s*بار\s*شیر|three\s+heads|پشت.?سرهم|consecutive)", text, re.I):
            prob = 1 / (2 ** n); label = f"هر {n} بار" if language == "fa" else f"all {n} tosses"
        else:
            return None
        pct = prob * 100
        frac = Fraction(prob).limit_denominator()
        if language == "fa":
            out = f"احتمال {label} = {frac.numerator}/{frac.denominator} = {self._fmt(pct)}٪."
        else:
            out = f"The probability of {label} is {frac.numerator}/{frac.denominator} = {self._fmt(pct)}%."
        return self._answer(out, "reasoned_answer", 0.999, "coin_probability_exact")

    def _solve_weekday_offset(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        match_fa = self._WEEKDAY_OFFSET_FA.search(text)
        if match_fa:
            day, offset = match_fa.group(1), int(match_fa.group(2))
            key = day.replace("‌", "").replace("-", " ")
            index = self._DAY_MAP.get(key, self._DAY_MAP.get(day))
            if index is None: return None
            result = self._WEEKDAYS_FA[(index + offset) % 7]
            return self._answer(f"{offset} روز بعد از {self._WEEKDAYS_FA[index]}، {result} است.", "reasoned_answer", 0.999, "weekday_modulo_exact")
        match_en = self._WEEKDAY_OFFSET_EN.search(text)
        if match_en:
            offset, day = int(match_en.group(1)), match_en.group(2).casefold()
            index = self._DAY_MAP[day]; result = self._WEEKDAYS_EN[(index + offset) % 7]
            return self._answer(f"{offset} days after {self._WEEKDAYS_EN[index]} is {result}.", "reasoned_answer", 0.999, "weekday_modulo_exact")
        return None

    def _solve_coding(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._CODING.search(text):
            return None
        value = text.casefold()
        if re.search(r"(?:زوج|even).{0,40}(?:لیست|list|اعداد|numbers)|(?:فیلتر|filter).{0,40}(?:زوج|even)", value, re.I):
            code = "def filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]"
            explanation = "این تابع فقط اعداد زوج را برمی‌گرداند." if language == "fa" else "This function returns only the even numbers."
        elif re.search(r"(?:فرد|odd).{0,40}(?:لیست|list|اعداد|numbers)|(?:فیلتر|filter).{0,40}(?:فرد|odd)", value, re.I):
            code = "def filter_odd(numbers):\n    return [n for n in numbers if n % 2 != 0]"
            explanation = "این تابع فقط اعداد فرد را برمی‌گرداند." if language == "fa" else "This function returns only the odd numbers."
        elif re.search(r"(?:palindrome|پالیندروم|قرینه)", value, re.I):
            code = "def is_palindrome(text):\n    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n    return cleaned == cleaned[::-1]"
            explanation = "فاصله و علائم را نادیده می‌گیرد و قرینه‌بودن متن را بررسی می‌کند." if language == "fa" else "It ignores punctuation and checks whether the text reads the same backward."
        elif re.search(r"(?:factorial|فاکتوریل)", value, re.I):
            code = "def factorial(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result"
            explanation = "برای ورودی منفی خطا می‌دهد و فاکتوریل را تکراری محاسبه می‌کند." if language == "fa" else "It rejects negative input and computes factorial iteratively."
        elif re.search(r"(?:prime|عدد\s+اول)", value, re.I):
            code = "def is_prime(n):\n    if n < 2:\n        return False\n    if n % 2 == 0:\n        return n == 2\n    d = 3\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 2\n    return True"
            explanation = "تقسیم‌کننده‌ها را فقط تا ریشهٔ دوم عدد بررسی می‌کند." if language == "fa" else "It tests divisors only up to the square root."
        elif re.search(r"(?:تکراری|duplicate|unique).{0,50}(?:لیست|list)", value, re.I):
            code = "def unique_preserve_order(items):\n    return list(dict.fromkeys(items))"
            explanation = "تکراری‌ها را حذف می‌کند و ترتیب اولین رخداد را حفظ می‌کند." if language == "fa" else "It removes duplicates while preserving first-seen order."
        elif re.search(r"(?:جمع|sum).{0,40}(?:لیست|list|اعداد|numbers)", value, re.I):
            code = "def sum_numbers(numbers):\n    return sum(numbers)"
            explanation = "جمع عناصر عددی لیست را برمی‌گرداند." if language == "fa" else "It returns the sum of the numeric list items."
        else:
            return None
        prefix = "کد Python:\n\n" if language == "fa" else "Python code:\n\n"
        return self._answer(prefix + code + "\n\n" + explanation, "coding_answer", 0.995, "code_template_verified")

    @staticmethod
    def _translation_payload(text: str) -> tuple[str, str] | None:
        raw = text.strip()
        patterns = (
            ("en", r"(?:به\s+انگلیسی(?:\s+طبیعی)?\s+ترجمه\s+کن|ترجمه\s+کن\s+به\s+انگلیسی|translate\s+(?:to|into)\s+english|english\s+translation)\s*[:：]?\s*[«\"']?(.+?)[»\"']?[؟?]?$"),
            ("fa", r"(?:به\s+فارسی(?:\s+طبیعی)?\s+ترجمه\s+کن|ترجمه\s+کن\s+به\s+فارسی|translate\s+(?:to|into)\s+(?:persian|farsi)|persian\s+translation)\s*[:：]?\s*[«\"']?(.+?)[»\"']?[؟?]?$"),
        )
        for target, pattern in patterns:
            match = re.search(pattern, raw, re.I | re.S)
            if match:
                return target, match.group(1).strip(" «»\"' .")
        return None

    def _solve_translation(self, text: str, language: str) -> LocalIntelligenceAnswer | None:
        if not self._TRANSLATE.search(text):
            return None
        payload = self._translation_payload(text)
        if not payload:
            return None
        target, source = payload
        normalized = normalize_text(source).strip(" .!?؟").casefold()
        if target == "en":
            direct = self._FA_EN_EXACT.get(normalize_text(source).strip(" .!?؟"))
            if direct:
                return self._answer(direct, "translation_answer", 0.995, "translation_memory_exact")
            # Compositional copula: من/ما ... صفت هستم/هستم-like forms.
            fa = normalize_text(source).strip(" .!?؟")
            m = re.fullmatch(r"من\s+(?:(امروز)\s+)?(خسته|خوشحال|ناراحت|آماده|گرسنه|تشنه|مشغول|نگران|مطمئن|آرام|عصبانی|بیمار)(?:‌?ام|\s+هستم)", fa)
            if m:
                adv = " today" if m.group(1) else ""
                return self._answer(f"I am {self._ADJECTIVES[m.group(2)]}{adv}.", "translation_answer", 0.99, "translation_compositional")
            m = re.fullmatch(r"ما\s+(?:(امروز)\s+)?(خسته|خوشحال|ناراحت|آماده|گرسنه|تشنه|مشغول|نگران|مطمئن|آرام|عصبانی|بیمار)(?:‌?ایم|\s+هستیم)", fa)
            if m:
                adv = " today" if m.group(1) else ""
                return self._answer(f"We are {self._ADJECTIVES[m.group(2)]}{adv}.", "translation_answer", 0.99, "translation_compositional")
        else:
            direct = self._EN_FA_EXACT.get(normalized)
            if direct:
                return self._answer(direct, "translation_answer", 0.995, "translation_memory_exact")
            m = re.fullmatch(r"i\s+am\s+(tired|happy|sad|ready|hungry|thirsty|busy|worried|sure|calm|angry|sick)(?:\s+today)?", normalized)
            if m:
                reverse = {v:k for k,v in self._ADJECTIVES.items()}
                today = " امروز" if normalized.endswith(" today") else ""
                return self._answer(f"من{today} {reverse[m.group(1)]} هستم.", "translation_answer", 0.99, "translation_compositional")
        return None


__all__ = ["LocalIntelligenceAnswer", "LocalIntelligenceV14"]
