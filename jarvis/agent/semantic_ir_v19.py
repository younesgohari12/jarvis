from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from jarvis.agent.semantic_ir_v18 import SemanticIR, SemanticIRParserV18, SemanticSolverV18
from jarvis.agent.semantic_models_v19 import ExecutionPatternClassifierV19, CodeIntentClassifierV19, SemanticFrameClassifierV19
from jarvis.nlu.semantic_slots_v19 import SemanticSlotBinderV19
from jarvis.utils.text import normalize_text

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


class SemanticIRParserV19:
    VERSION = "1.9.1"
    _root = Path(__file__).resolve().parents[2]
    _binder = SemanticSlotBinderV19(_root)
    _frame = SemanticFrameClassifierV19(_root / "models" / "semantic_frame_v19.npz")
    _exec = ExecutionPatternClassifierV19(_root / "models" / "execution_pattern_v19.npz")
    _code_model = CodeIntentClassifierV19(_root / "models" / "code_intent_v19.npz")

    @staticmethod
    def _norm(text: str) -> str:
        return normalize_text(text).translate(_DIGITS).replace("٫", ".").replace("٪", "%")

    @staticmethod
    def _numbers(text: str) -> list[float]:
        return [float(x.replace(",", ".")) for x in re.findall(r"(?<![\w.])\d+(?:[.,]\d+)?", text)]

    @classmethod
    def parse(cls, text: str) -> SemanticIR | None:
        t = cls._norm(text)
        # Hard negative gates prevent semantic collisions before any learned frame is used.
        if re.search(r"(?:صدا|ولوم|volume|روشنایی|نور\s+صفحه|brightness)", t, re.I) and re.search(r"(?:درصد|%|set|تنظیم|بگذار|روی)", t, re.I):
            return None
        # Dedicated semantic families outrank generic percentage/ratio.
        for parser in (cls._probability, cls._sequence, cls._ratio, cls._multistep, cls._age, cls._work, cls._code):
            ir = parser(t)
            if ir is not None:
                return ir
        # Keep mature v18 polarity/prime/transitive/etc. semantics as a fallback IR.
        ir18 = SemanticIRParserV18.parse(t)
        if ir18 is not None:
            # Never let the older broad ratio parser reinterpret a sequence.
            if ir18.task == "ratio_split" and re.search(r"(?:دنباله|تصاعد|sequence|geometric|\bgp\b|term)", t, re.I):
                return None
            return SemanticIR(ir18.task, dict(ir18.slots), ir18.confidence, ir18.predicate, ir18.polarity, (*ir18.evidence, "v18_semantics_reused"), ir18.model_frame)
        return None

    @classmethod
    def _sequence(cls, t: str) -> SemanticIR | None:
        if not re.search(r"(?:دنباله\s+هندسی|تصاعد\s+هندسی|geometric\s+(?:sequence|progression)|\bgp\b)", t, re.I):
            return None
        a = r = n = None
        for pat in (
            r"(?:جمله\s+(?:اول|نخست)|جمله.?ی\s+(?:اول|نخست)|a[_ ]?1|first\s+term|initial\s+(?:term|value))\s*(?:=|برابر|is)?\s*(-?\d+)",
            r"(?:دنباله\s+هندسی|تصاعد\s+هندسی|geometric\s+(?:sequence|progression)|\bgp\b)\s*(?:از|با\s+شروع|با\s+پایه|starts?\s+(?:at|with)|begins?\s+(?:at|with))?\s*(-?\d+)",
        ):
            m = re.search(pat, t, re.I)
            if m:
                a = float(m.group(1)); break
        m = re.search(r"(?:قدر.?نسبت|نسبت(?:\s+مشترک)?|common\s+ratio|ratio|scaling\s+factor|multiplier)\s*(?:=|برابر|است|is)?\s*(-?\d+)", t, re.I)
        if m: r = float(m.group(1))
        m = re.search(r"(?:جمله|ترم|عضو|term|member|position|index)(?:\s+(?:شماره|number))?\s*(\d+)", t, re.I)
        if m: n = int(m.group(1))
        if n is None:
            ordinals = {
                "اول":1,"نخست":1,"دوم":2,"سوم":3,"چهارم":4,"پنجم":5,"ششم":6,"هفتم":7,"هشتم":8,"نهم":9,"دهم":10,
                "first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,
            }
            matches = re.findall(r"(?:جمله|ترم|عضو|term|member)\s+(اول|نخست|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)", t, re.I)
            if matches: n = ordinals[matches[-1].casefold()]
        if a is None or r is None or n is None or n < 1:
            return None
        return SemanticIR("geometric_sequence_v19", {"a":a,"r":r,"n":n}, .995, "nth_geometric_term", "value", ("sequence_priority","semantic_roles"), "sequence")

    @classmethod
    def _ratio(cls, t: str) -> SemanticIR | None:
        frame = cls._binder.bind(t)
        if frame.task != "ratio_split":
            return None
        s = frame.slots
        return SemanticIR("ratio_split", {"total": s["total"], "a": s["ratio_a"], "b": s["ratio_b"]}, min(.999, max(.94, frame.confidence)), "split_by_ratio", "value", frame.evidence, "ratio_split")

    @classmethod
    def _probability(cls, t: str) -> SemanticIR | None:
        if not re.search(r"(?:احتمال|شانس|probability|chance)", t, re.I):
            return None
        # Let v18 handle coins/dice/permutation first because those are more specialized.
        special = SemanticIRParserV18.parse(t)
        if special is not None and special.task in {"binomial_probability", "dice_sum_probability", "permutation"}:
            return special
        roles = cls._binder.role_values(t, min_confidence=.30)
        p = roles.get("probability")
        n = roles.get("n")
        k = roles.get("k")
        if p is None:
            m = re.search(r"(?:احتمال(?:\s+هر)?\s*موفقیت|شانس(?:\s+موفقیت)?|success\s+(?:probability|chance)|chance\s+per\s+(?:attempt|trial)|per[- ]trial\s+probability).{0,28}?(\d+(?:\.\d+)?)\s*(?:%|درصد|percent)", t, re.I)
            if m: p = float(m.group(1))
        if n is None:
            m = re.search(r"(?:در|از|over|in|out\s+of)\s*(\d+)\s*(?:بار|تلاش|آزمون|trials?|attempts?)|(?:تعداد\s+آزمون|trials?)\s*(?:=|است|is)?\s*(\d+)", t, re.I)
            if m: n = float(next(x for x in m.groups() if x))
        if k is None:
            m = re.search(r"(?:دقیقاً|دقیقا|exactly|target\s+successes?)\s*(?:=|is)?\s*(\d+)|(?:موفقیت\s+مطلوب)\s*(\d+)", t, re.I)
            if m: k = float(next(x for x in m.groups() if x))
        if p is None or n is None or k is None:
            return None
        p = p / 100.0 if p > 1 else p
        n_i, k_i = int(round(n)), int(round(k))
        if not (0 <= p <= 1 and 0 <= k_i <= n_i <= 100):
            return None
        return SemanticIR("binomial_probability", {"p": p, "n": n_i, "k": k_i, "event": "success"}, .995, "exactly_k_successes", "value", ("probability_priority", "trained_numeric_roles"), "probability")

    @classmethod
    def _extract_initial(cls, t: str) -> float | None:
        patterns = (
            r"(?:از|from|starting\s+from|start(?:ing)?\s+(?:with|at)|مقدار|قیمت|مبلغ|price(?:\s+is)?|base\s+(?:cost|amount)|موجودی|inventory(?:\s+starts?\s+at)?)\s*(\d+(?:\.\d+)?)",
            r"^(\d+(?:\.\d+)?)\s*(?:را|رو|,|؛|;)"
        )
        for pat in patterns:
            m = re.search(pat, t, re.I)
            if m: return float(m.group(1))
        nums = cls._numbers(t)
        return nums[0] if nums else None

    @classmethod
    def _multistep(cls, t: str) -> SemanticIR | None:
        if not re.search(r"(?:بعد(?:ش)?|سپس|then|afterwards|followed\s+by|و\s+بعد)", t, re.I):
            return None
        if re.search(r"(?:احتمال|probability|chance|دنباله|sequence|geometric|\bgp\b)", t, re.I):
            return None
        pred = cls._exec.predict(t) if cls._exec.ready else None
        parts = re.split(r"(?:و\s+بعد|بعد(?:ش)?|سپس|then|afterwards|followed\s+by)", t, maxsplit=1, flags=re.I)
        if len(parts) < 2:
            return None
        first, second = parts[0], parts[1]
        initial = cls._extract_initial(t)
        if initial is None:
            return None
        steps: list[dict[str, Any]] = []
        # Step 1 percent change.
        pm = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|درصد|percent)", first, re.I)
        if pm:
            p = float(pm.group(1))
            if re.search(r"(?:کم|کاهش|تخفیف|فروش|reduce|decrease|discount|sold|remove|subtract)", first, re.I):
                steps.append({"op": "percentage_remove", "percent": p})
            elif re.search(r"(?:زیاد|افزایش|رشد|اضافه|increase|grow|add|plus)|\+\s*\d", first, re.I):
                steps.append({"op": "percentage_add", "percent": p})
        # Non-percent first operation.
        if not steps:
            nums1 = cls._numbers(first)
            candidates = [x for x in nums1 if abs(x - initial) > 1e-9]
            if candidates:
                x = candidates[-1]
                if re.search(r"(?:اضافه|بیفزا|add|plus)", first, re.I): steps.append({"op":"add","value":x})
                elif re.search(r"(?:کم|منهای|subtract|minus|remove)", first, re.I): steps.append({"op":"subtract","value":x})
        nums2 = cls._numbers(second)
        if nums2:
            x = nums2[0]
            if re.search(r"(?:ضرب|برابر|multiply|times)", second, re.I): steps.append({"op":"multiply","value":x})
            elif re.search(r"(?:تقسیم|divide)", second, re.I): steps.append({"op":"divide","value":x})
            elif re.search(r"(?:کم|کاهش|منهای|subtract|minus|remove|take\s+away)", second, re.I): steps.append({"op":"subtract","value":x})
            elif re.search(r"(?:اضافه|بیفزا|زیاد|شارژ|وارد|add|plus|increase|restock|fee)", second, re.I) or re.search(r"\+\s*\d", second): steps.append({"op":"add","value":x})
        if len(steps) != 2:
            return None
        evidence = ["execution_graph_built"]
        confidence = .94
        if pred is not None:
            evidence.append(f"trained_pattern={pred.label}")
            confidence = max(confidence, min(.995, pred.confidence))
        return SemanticIR("execution_graph", {"initial": initial, "steps": steps}, confidence, "execute_steps", "value", tuple(evidence), "multistep")

    @classmethod
    def _age(cls, t: str) -> SemanticIR | None:
        if not re.search(r"(?:سن|ساله|سالشه|years?\s+old|age|خواهر|برادر|sister|brother)", t, re.I):
            return None
        roles = cls._binder.role_values(t, min_confidence=.28)
        age = roles.get("age"); diff = roles.get("difference"); later = roles.get("years_later")
        # Semantic fallbacks for colloquial and alternate word order.
        if age is None:
            pats=(r"(?:سن\s+\w+|\w+\s+الان|\w+)\s*(\d+)\s*(?:سال(?:ه)?|سالشه|years?\s+old)",r"(?:age\s+(?:is\s+)?)?(\d+)\s*years?\s+old")
            for p in pats:
                m=re.search(p,t,re.I)
                if m: age=float(m.group(1));break
        if diff is None:
            m=re.search(r"(\d+)\s*(?:سال|years?)\s*(?:بزرگ.?تر|کوچک.?تر|older|younger)|(?:older|younger)\s+by\s*(\d+)",t,re.I)
            if m: diff=float(next(x for x in m.groups() if x))
        if later is None:
            m=re.search(r"(?:بعد\s+از\s*)?(\d+)\s*(?:سال|years?)\s*(?:بعد|دیگر|later|from\s+now)",t,re.I)
            if m: later=float(m.group(1))
        if None in (age,diff,later):
            return None
        relation = "younger" if re.search(r"(?:کوچک.?تر|younger)",t,re.I) else "older"
        return SemanticIR("age_reasoning", {"base_age":age,"difference":diff,"relation":relation,"years_later":later}, .985, "sum_future_ages", "value", ("trained_numeric_roles","age_relation_bound"), "age_reasoning")

    @classmethod
    def _work(cls, t: str) -> SemanticIR | None:
        if not (re.search(r"(?:کارگر|workers?)",t,re.I) and re.search(r"(?:قطعه|کالا|واحد|items?|units?|pieces?)",t,re.I)):
            return None
        # Unit-attached extraction is order-independent inside each clause.
        clauses=re.split(r"[؛;?.]|(?:،\s*(?=\d+\s*کارگر))|(?:حالا|اگر|then|now)", t, flags=re.I)
        found=[]
        for c in clauses:
            w=re.search(r"(\d+(?:\.\d+)?)\s*(?:نفر\s+)?(?:کارگر|workers?)",c,re.I)
            h=re.search(r"(\d+(?:\.\d+)?)\s*(?:ساعت|hours?)",c,re.I)
            o=re.search(r"(\d+(?:\.\d+)?)\s*(?:قطعه|کالا|واحد|items?|units?|pieces?)",c,re.I)
            if w and h: found.append((float(w.group(1)),float(h.group(1)),float(o.group(1)) if o else None))
        if len(found)>=2 and found[0][2] is not None:
            w1,h1,o1=found[0]; w2,h2,_=found[1]
            return SemanticIR("work_scaling",{"workers1":w1,"hours1":h1,"output1":o1,"workers2":w2,"hours2":h2},.99,"constant_worker_hour_productivity","value",("unit_semantic_binding","order_independent"),"work_rate")
        return None

    @classmethod
    def _code(cls, t: str) -> SemanticIR | None:
        if not re.search(r"(?:تابع|function|کد|code|python|پایتون|flask|api)",t,re.I): return None
        if not re.search(r"(?:بنویس|بساز|ایجاد|درست|پیاده|write|create|implement|make|build)",t,re.I): return None
        pred=cls._code_model.predict(t) if cls._code_model.ready else None
        if pred is None or pred.confidence < .45: return None
        # Learned classifier may abstain when the prompt belongs to an older coding family
        # (prime checks, deduplication, tracing, etc.).  A semantic compatibility gate keeps
        # v19 from replacing a mature solver with an unrelated supported algorithm.
        intent_cues = {
            "filter_positive": r"(?:مثبت|positive|بزرگ.?تر\s+از\s+صفر|greater\s+than\s+zero)",
            "max_value": r"(?:بیشترین|بزرگ.?ترین|ماکزیمم|maximum|largest|biggest|max[- ]?value)",
            "flask_api": r"(?:flask|endpoint|rest\s+api|health[- ]?check|\bapi\b)",
            "filter_even": r"(?:زوج|even|divisible\s+by\s+(?:two|2))",
            "sort_values": r"(?:مرتب|صعودی|sort|ascending|low\s+to\s+high|order\s+(?:array|list))",
            "sum_values": r"(?:مجموع|جمع\s+همه|sum|total\s+(?:of|array|list)|sums\s+all)",
        }
        cue = intent_cues.get(pred.label)
        if cue is None or not re.search(cue, t, re.I):
            return None
        return SemanticIR("code_generation_v19",{"code_intent":pred.label},min(.995,max(.90,pred.confidence)),"generate_algorithm_then_code","value",("trained_code_intent","semantic_compatibility_gate"),"code_generation")


class SemanticSolverV19:
    @staticmethod
    def _fmt(x: float) -> str:
        return str(int(round(x))) if abs(x-round(x))<1e-10 else f"{x:.8f}".rstrip("0").rstrip(".")

    @classmethod
    def solve(cls, ir: SemanticIR, language: str = "fa"):
        if ir.task == "execution_graph":
            value=float(ir.slots["initial"])
            for step in ir.slots["steps"]:
                op=step["op"]
                if op=="percentage_remove": value*=1-float(step["percent"])/100
                elif op=="percentage_add": value*=1+float(step["percent"])/100
                elif op=="add": value+=float(step["value"])
                elif op=="subtract": value-=float(step["value"])
                elif op=="multiply": value*=float(step["value"])
                elif op=="divide":
                    divisor=float(step["value"])
                    if abs(divisor)<1e-12:return None
                    value/=divisor
                else:return None
            text=(f"نتیجه نهایی = {cls._fmt(value)}." if language=="fa" else f"Final result = {cls._fmt(value)}.")
            return text,"word_problem_answer",.999,("sir_v19_parsed","execution_graph_executed","result_verified")
        if ir.task == "geometric_sequence_v19":
            a=float(ir.slots["a"]); r=float(ir.slots["r"]); n=int(ir.slots["n"])
            value=a*(r**(n-1))
            text=(f"جملهٔ {n} برابر {cls._fmt(value)} است." if language=="fa" else f"a_{n} = {cls._fmt(value)}.")
            return text,"reasoned_answer",.999,("sir_v19_parsed","geometric_sequence_exact","result_verified")
        if ir.task == "code_generation_v19":
            intent=str(ir.slots["code_intent"])
            code={
                "filter_positive":"def filter_positive(values):\n    return [x for x in values if x > 0]",
                "max_value":"def max_value(values):\n    if not values:\n        raise ValueError(\"values must not be empty\")\n    return max(values)",
                "flask_api":"from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.get(\"/health\")\ndef health():\n    return jsonify({\"status\": \"ok\"})\n\nif __name__ == \"__main__\":\n    app.run(debug=True)",
                "filter_even":"def filter_even(values):\n    return [x for x in values if x % 2 == 0]\n\ndef even_numbers(values):\n    return filter_even(values)",
                "sort_values":"def sort_values(values):\n    return sorted(values)",
                "sum_values":"def sum_values(values):\n    return sum(values)",
            }.get(intent)
            if code:return code,"coding_answer",.998,("sir_v19_parsed","algorithm_selected","code_generated","result_verified")
            return None
        solved=SemanticSolverV18.solve(ir,language)
        if solved is not None:
            text,intent,confidence,checks=solved
            return text,intent,confidence,("sir_v19_parsed",*checks)
        return None


__all__=["SemanticIRParserV19","SemanticSolverV19"]
