from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis.agent.semantic_models_v19 import (
    SemanticFrameClassifierV19,
    NumericRoleTaggerV19,
    ExecutionPatternClassifierV19,
    CodeIntentClassifierV19,
)

SEED = 19092026
R = random.Random(SEED)
VERSION = "dataset_v019_real"
DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
FRAME_SIZE = 65536
NUM_SIZE = 65536
EXEC_SIZE = 32768
CODE_SIZE = 32768


def fmt_fa(n: int) -> str:
    s = str(n)
    return s.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def distinct_values(specs: list[tuple[str, int, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    used: set[int] = set()
    for name, lo, hi in specs:
        for _ in range(100):
            v = R.randint(lo, hi)
            if v not in used:
                out[name] = v; used.add(v); break
        else:
            raise RuntimeError("could not make distinct values")
    return out


def template_split(template_id: str) -> str:
    # deterministic template-level holdout: no template id crosses splits
    import hashlib
    v = int(hashlib.sha256((template_id + str(SEED)).encode()).hexdigest()[:8], 16) % 100
    return "train" if v < 70 else "validation" if v < 85 else "test"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")

def assign_stratified_splits(rows: list[dict], group_key: str) -> None:
    groups=defaultdict(set)
    for r in rows: groups[str(r[group_key])].add(str(r["template_id"]))
    mapping={}
    for group,tids in groups.items():
        ordered=sorted(tids)
        for i,tid in enumerate(ordered):
            frac=(i+0.5)/max(1,len(ordered))
            mapping[(group,tid)]="train" if frac<=.70 else "validation" if frac<=.85 else "test"
    for r in rows: r["split"]=mapping[(str(r[group_key]),str(r["template_id"]))]


def build_numeric_rows(target: int = 24000) -> list[dict]:
    templates: list[tuple[str, str, str, list[str]]] = []
    # Work-rate: deliberately change number order and syntax.
    fa_work = [
        "{workers} کارگر در {hours} ساعت {output} قطعه تولید می‌کنند",
        "در {hours} ساعت، {workers} کارگر {output} کالا می‌سازند",
        "تولید {output} واحد با {workers} کارگر {hours} ساعت زمان می‌برد",
        "اگر مدت کار {hours} ساعت و تعداد کارگرها {workers} نفر باشد، خروجی {output} قطعه است",
        "خروجی {output} است؛ این مقدار را {workers} کارگر طی {hours} ساعت ساخته‌اند",
        "طی {hours} ساعت {output} قطعه توسط {workers} کارگر تولید شد",
        "{output} کالا حاصل کار {workers} کارگر در مدت {hours} ساعت است",
        "با {workers} نیروی کار و زمان {hours} ساعت، تولید به {output} واحد رسید",
        "زمان {hours} ساعت بود، خروجی {output} قطعه و تعداد کارگر {workers} نفر",
        "{workers} نفر کارگر، {output} قطعه را ظرف {hours} ساعت تولید کردند",
    ]
    en_work = [
        "{workers} workers produce {output} units in {hours} hours",
        "In {hours} hours, {workers} workers make {output} items",
        "Producing {output} pieces takes {hours} hours with {workers} workers",
        "Output is {output} units when {workers} workers work for {hours} hours",
        "During {hours} hours, an output of {output} was made by {workers} workers",
        "With {workers} workers and {hours} hours, production reaches {output} units",
        "The job used {hours} hours, {workers} workers, and produced {output} pieces",
        "{output} items were completed in {hours} hours by {workers} workers",
        "Worker count is {workers}; time is {hours} hours; output is {output}",
        "The team made {output} units using {workers} workers over {hours} hours",
    ]
    extra_work = [
        "تعداد کارگر {workers} نفر است؛ در مدت {hours} ساعت، {output} واحد خروجی داریم",
        "{hours} ساعت کار با تیم {workers} نفره برابر با تولید {output} قطعه شد",
        "برای ساخت {output} قطعه، {workers} کارگر مجموعاً {hours} ساعت کار کردند",
        "کارگاه با {workers} کارگر بعد از {hours} ساعت به تولید {output} واحد رسید",
        "{output} واحد محصول در بازه {hours} ساعته توسط {workers} کارگر ساخته شد",
        "اگر {workers} کارگر داشته باشیم و {hours} ساعت کار کنند، خروجی ثبت‌شده {output} است",
        "تیم شامل {workers} کارگر بود؛ زمان {hours} ساعت و محصول نهایی {output} قطعه",
        "در یک شیفت {hours} ساعته، گروه {workers} نفره {output} کالا تولید کرد",
        "با زمان کاری {hours} ساعت، {output} قطعه از {workers} کارگر به دست آمد",
        "نرخ را از این داده بگیر: کارگر {workers}، ساعت {hours}، تولید {output}",
        "A crew of {workers} workers completed {output} pieces after {hours} hours",
        "The production record shows {output} units, {hours} working hours, and {workers} workers",
        "For {output} finished items, the team used {workers} workers for {hours} hours",
        "A {workers}-worker team working {hours} hours produced {output} units",
        "After {hours} hours of work, the {workers} workers had made {output} pieces",
        "Production reached {output}; staffing was {workers} workers and duration {hours} hours",
        "Given output {output}, worker count {workers}, and time {hours} hours, find the rate",
        "The shift lasted {hours} hours with {workers} workers and yielded {output} units",
        "There were {workers} workers; they made {output} items within {hours} hours",
        "Over a period of {hours} hours, production by {workers} workers totaled {output} pieces",
    ]
    for i, t in enumerate(fa_work + en_work + extra_work): templates.append(("work", f"work_{i:02d}", t, ["workers","hours","output"]))

    # Ratio split: strict split semantics, multiple positions and languages.
    split_verbs_fa = ["تقسیم کن", "پخش کن", "سهم‌بندی کن", "بین دو نفر تقسیم کن", "به دو سهم تقسیم کن"]
    ratio_forms_fa = [
        "مبلغ {total} را با نسبت {ratio_a} به {ratio_b} {verb}",
        "با نسبت {ratio_a}:{ratio_b}، {total} واحد را {verb}",
        "{total} را {verb} طوری که نسبت سهم‌ها {ratio_a} به {ratio_b} باشد",
        "نسبت دو سهم {ratio_a} به {ratio_b} است؛ عدد {total} را {verb}",
        "برای تقسیم {total}، نسبت سهم اول به دوم را {ratio_a}:{ratio_b} بگیر و {verb}",
    ]
    split_verbs_en = ["split", "divide", "distribute", "allocate between two people"]
    ratio_forms_en = [
        "{verb} {total} in the ratio {ratio_a} to {ratio_b}",
        "Using a {ratio_a}:{ratio_b} ratio, {verb} the total {total}",
        "The share ratio is {ratio_a} to {ratio_b}; {verb} {total}",
        "Take {total} and {verb} it so the two shares are in ratio {ratio_a}:{ratio_b}",
        "For total {total}, {verb} it with first-to-second ratio {ratio_a} to {ratio_b}",
    ]
    tid = 0
    for form in ratio_forms_fa:
        for verb in split_verbs_fa:
            templates.append(("ratio", f"ratio_fa_{tid:02d}", form.replace("{verb}", verb), ["total","ratio_a","ratio_b"])); tid += 1
    for form in ratio_forms_en:
        for verb in split_verbs_en:
            templates.append(("ratio", f"ratio_en_{tid:02d}", form.replace("{verb}", verb), ["total","ratio_a","ratio_b"])); tid += 1

    # Money/percentage with both orders.
    money_forms = [
        "مبلغ {money} با تخفیف {percentage} درصد",
        "{percentage} درصد از قیمت {money} کم کن",
        "قیمت اولیه {money} است و درصد تخفیف {percentage}",
        "با {percentage}٪ تخفیف، مبلغ پایه {money} است",
        "درصد تخفیف {percentage} است؛ قیمت قبل از تخفیف {money}",
        "A {percentage} percent discount applies to price {money}",
        "Price {money} has a {percentage}% discount",
        "Discount rate is {percentage} percent on an amount of {money}",
        "The base amount is {money}; reduce it by {percentage} percent",
        "Apply {percentage}% off to the {money} price",
    ]
    money_forms += [
        "قیمت کالا {money} است؛ {percentage} درصد از آن تخفیف بده",
        "روی مبلغ {money} تخفیفی به اندازه {percentage}٪ اعمال می‌شود",
        "مبلغ قبل از تخفیف {money} و نرخ کاهش {percentage} درصد است",
        "اگر قیمت {money} باشد و {percentage} درصد کم شود، مبلغ نهایی را حساب کن",
        "از قیمت پایه {money} باید {percentage}٪ کسر شود",
        "برای مبلغ {money} نرخ تخفیف را {percentage} درصد در نظر بگیر",
        "هزینه اولیه {money} است و قرار است {percentage} درصد کاهش پیدا کند",
        "{money} تومان قیمت داریم؛ تخفیف فروش {percentage} درصد است",
        "با نرخ تخفیف {percentage}٪ روی عدد {money} محاسبه کن",
        "عدد مالی {money} را با کاهش درصدی {percentage} پردازش کن",
        "The original price is {money} and the markdown is {percentage} percent",
        "Take an amount of {money} and apply a {percentage}% reduction",
        "The item costs {money} before a {percentage} percent discount",
        "A markdown rate of {percentage}% is applied to {money}",
        "Use base price {money} with discount percentage {percentage}",
        "Starting cost {money}; percentage off is {percentage}",
        "Calculate a {percentage}% reduction on the amount {money}",
        "The pre-discount amount equals {money}, while discount rate equals {percentage}%",
        "For an invoice amount of {money}, apply {percentage} percent off",
        "Amount {money} is subject to a {percentage}% discount",
    ]
    for i, t in enumerate(money_forms): templates.append(("money_percent", f"money_{i:02d}", t, ["money","percentage"]))

    # Distance/time/speed cues; some rows contain only 2 roles, some 3.
    speed_forms = [
        "مسافت {distance} کیلومتر در زمان {hours} ساعت طی شد",
        "در {hours} ساعت، {distance} کیلومتر حرکت کرد",
        "فاصله {distance} کیلومتر و مدت سفر {hours} ساعت بود",
        "زمان سفر {hours} ساعت بود و مسافت {distance} کیلومتر",
        "{distance} کیلومتر را طی {hours} ساعت رفت",
        "Distance is {distance} km and time is {hours} hours",
        "In {hours} hours the vehicle covered {distance} km",
        "Travel time {hours} hours; distance {distance} km",
        "It covered {distance} kilometers over {hours} hours",
        "The trip lasted {hours} hours for a distance of {distance} km",
    ]
    speed_forms += [
        "خودرو در مدت {hours} ساعت مسافت {distance} کیلومتر را پیمود",
        "مدت حرکت {hours} ساعت و طول مسیر {distance} کیلومتر ثبت شد",
        "برای مسیر {distance} کیلومتری، زمان سفر {hours} ساعت بود",
        "{hours} ساعت زمان صرف شد تا {distance} کیلومتر طی شود",
        "فاصله طی‌شده {distance} کیلومتر است؛ زمان مصرف‌شده {hours} ساعت",
        "حرکت به اندازه {distance} کیلومتر در بازه {hours} ساعته انجام شد",
        "سفر {hours} ساعت طول کشید و طول آن {distance} کیلومتر بود",
        "در بازه زمانی {hours} ساعت، پیمایش {distance} کیلومتر بود",
        "زمان={hours} ساعت و فاصله={distance} کیلومتر",
        "برای محاسبه سرعت: {hours} ساعت زمان و {distance} کیلومتر مسافت",
        "The journey covered {distance} km in a duration of {hours} hours",
        "A distance of {distance} kilometers was traveled over {hours} hours",
        "The elapsed time was {hours} hours for {distance} km",
        "Use {distance} km and {hours} hours to compute average speed",
        "Travel duration: {hours} hours; route length: {distance} km",
        "It took {hours} hours to travel the {distance}-km route",
        "The vehicle moved {distance} kilometers during a {hours}-hour interval",
        "Recorded distance is {distance} km while recorded time is {hours} hours",
        "For this trip, time equals {hours} h and distance equals {distance} km",
        "The route length is {distance} km and the trip lasted {hours} hours",
    ]
    for i, t in enumerate(speed_forms): templates.append(("distance_time", f"dist_{i:02d}", t, ["distance","hours"]))

    # Age: order-independent base/difference/future offset.
    age_forms = [
        "علی {age} سال دارد، خواهرش {difference} سال بزرگ‌تر است؛ {years_later} سال بعد مجموع سن؟",
        "{years_later} سال دیگر مجموع سن را می‌خواهیم؛ علی الان {age} سالشه و خواهرش {difference} سال بزرگتره",
        "سن علی {age} است و اختلاف سن با خواهرش {difference} سال؛ بعد از {years_later} سال مجموع چند می‌شود؟",
        "خواهر علی {difference} سال از او بزرگ‌تر است؛ علی {age} ساله است؛ {years_later} سال بعد جمع سن‌ها؟",
        "الان علی {age} سال دارد؛ {years_later} سال بعد، با خواهری که {difference} سال بزرگ‌تر است مجموع سنشان؟",
        "Tom is {age} years old; his sister is {difference} years older. What is their total age {years_later} years later?",
        "In {years_later} years, find their age sum: Tom is {age} now and his sister is {difference} years older",
        "Tom's age is {age}; the sister age gap is {difference}; after {years_later} years find the total",
        "His sister is {difference} years older than Tom, who is {age}; what is the sum {years_later} years from now?",
        "Tom has reached age {age}; his sister is older by {difference} years; total after {years_later} years?",
    ]
    age_forms += [
        "علی الان {age} ساله است؛ خواهر او {difference} سال بزرگ‌تر است؛ پس از {years_later} سال مجموع سن‌ها؟",
        "{difference} سال اختلاف سن دارند و خواهر بزرگ‌تر است؛ علی {age} سال دارد؛ {years_later} سال بعد جمع؟",
        "برای {years_later} سال آینده حساب کن: سن فعلی علی {age} و اختلاف با خواهر {difference} سال",
        "سن پایه {age} است، نفر دوم {difference} سال بزرگ‌تر؛ بعد از {years_later} سال مجموع را بده",
        "علی به سن {age} رسیده و خواهرش {difference} سال از او بزرگ‌تر است؛ {years_later} سال دیگر جمع سن؟",
        "خواهر {difference} سال بزرگ‌تر از علی است؛ اگر علی اکنون {age} سال دارد، {years_later} سال بعد مجموع؟",
        "الان سن علی {age} است. اختلاف با خواهر {difference} و افق زمانی {years_later} سال بعد است",
        "با سن فعلی {age} برای علی و اختلاف {difference} سال، مجموع سن دو نفر در {years_later} سال آینده؟",
        "{years_later} سال بعد را در نظر بگیر؛ علی امروز {age} و خواهرش {difference} سال بزرگ‌تر",
        "سن علی الان {age}؛ خواهر +{difference} سال؛ آینده +{years_later} سال؛ مجموع؟",
        "Tom's current age is {age}, his sister is older by {difference} years; find their sum after {years_later} years",
        "Tom has age {age}; with a sister {difference} years older, what is the total {years_later} years later?",
        "Look {years_later} years ahead: Tom is {age} today and his sister is older by {difference}",
        "Current age {age}, sibling gap {difference} years older, future offset {years_later}; find the sum",
        "Tom is now {age}; his sister exceeds his age by {difference}; total ages in {years_later} years?",
        "His sister is older by {difference}. Tom is currently {age}. Add their ages after {years_later} years",
        "At age {age}, Tom is {difference} years younger than his sister; what is their total after {years_later} years?",
        "Tom reached {age}; sister age difference is +{difference}; compute combined age {years_later} years from now",
        "Given Tom={age}, sister older by {difference}, and future={years_later} years, calculate the total",
        "In {years_later} years from now, sum ages when Tom is {age} and sister is {difference} years older",
    ]
    for i, t in enumerate(age_forms): templates.append(("age", f"age_{i:02d}", t, ["age","difference","years_later"]))

    # Probability roles, designed to collide with generic percentage unless semantics are learned.
    prob_forms = [
        "احتمال موفقیت {probability} درصد است؛ در {n} بار دقیقاً {k} موفقیت چه احتمالی دارد؟",
        "در {n} تلاش، احتمال هر موفقیت {probability}٪ است؛ احتمال دقیقاً {k} موفقیت؟",
        "برای رویدادی با شانس {probability} درصد، از {n} آزمون دقیقاً {k} بار موفقیت",
        "دقیقاً {k} موفقیت در {n} بار وقتی احتمال موفقیت {probability} درصد است",
        "شانس موفقیت {probability}٪؛ تعداد آزمون {n} و موفقیت مطلوب {k}",
        "Success probability is {probability}%; in {n} trials, probability of exactly {k} successes?",
        "Exactly {k} successes out of {n} attempts with success chance {probability} percent",
        "With a {probability}% success rate, find P(exactly {k} in {n} trials)",
        "There are {n} trials and target successes {k}; per-trial probability is {probability}%",
        "Chance per attempt {probability} percent; over {n} attempts, exactly {k} successes",
    ]
    prob_forms += [
        "اگر شانس موفقیت هر بار {probability}٪ باشد، در {n} کوشش احتمال دقیقاً {k} موفقیت؟",
        "در {n} آزمایش مستقل با احتمال موفقیت {probability} درصد، دقیقاً {k} موفقیت را حساب کن",
        "احتمال پایه موفقیت {probability}٪ است؛ تعداد تکرار {n} و تعداد موفقیت هدف {k}",
        "برای {n} بار تکرار و نرخ موفقیت {probability} درصد، احتمال رخ دادن دقیقاً {k} موفقیت؟",
        "شانس هر آزمون {probability}٪؛ می‌خواهیم از {n} بار، دقیقاً {k} بار موفق شود",
        "در مدل دوجمله‌ای با p={probability} درصد، n={n} و k={k} را حساب کن",
        "نرخ موفقیت {probability} درصد و تعداد تلاش‌ها {n} است؛ احتمال {k} موفقیت دقیق؟",
        "از {n} کوشش، دقیقاً {k} موفقیت با شانس هر کوشش {probability}٪",
        "برای رویداد مستقل با probability {probability}٪، در {n} بار target={k}",
        "احتمال هر موفقیت {probability} درصد؛ n برابر {n} و k برابر {k}",
        "With per-trial success probability {probability}%, what is the chance of exactly {k} successes in {n} trials?",
        "For {n} independent attempts at {probability}% success each, calculate exactly {k} successes",
        "Use binomial parameters p={probability} percent, n={n}, k={k}",
        "The success chance per trial is {probability}%; target {k} successes across {n} trials",
        "Across {n} attempts with a {probability}% rate, find the probability of {k} exact successes",
        "Per attempt chance is {probability} percent; total attempts {n}; desired successes {k}",
        "Given n={n}, k={k}, and success probability {probability}%, compute the binomial probability",
        "In {n} repeated trials, each succeeds with chance {probability}%; exactly {k} should succeed",
        "Target exactly {k} successes from {n} trials when p is {probability}%",
        "A trial has {probability}% success probability; repeat it {n} times and ask for exactly {k} successes",
    ]
    for i, t in enumerate(prob_forms): templates.append(("probability", f"prob_{i:02d}", t, ["probability","n","k"]))

    # Generic count/index to prevent every number from becoming answer/percentage.
    generic_forms = [
        "از {count} مورد، مورد شماره {index} را انتخاب کن",
        "تعداد کل {count} است و شاخص موردنظر {index}",
        "در لیست {count} عضوی، خانه {index} را بررسی کن",
        "Select item {index} from a list of {count} items",
        "The list has {count} entries; inspect index {index}",
        "Among {count} records, use position {index}",
    ]
    generic_forms += [
        "کل موارد {count} تاست؛ شماره {index} را بردار", "در میان {count} گزینه، گزینه {index} را بررسی کن",
        "فهرست {count} مورد دارد و موقعیت هدف {index} است", "از مجموعه {count} عضوی، آیتم با شاخص {index} را انتخاب کن",
        "تعداد رکوردها {count} و شماره رکورد موردنظر {index}", "بین {count} نتیجه، نتیجه شماره {index} را نشان بده",
        "لیست شامل {count} عنصر است؛ عنصر جایگاه {index} را بخوان", "اگر {count} آیتم داریم، آیتم {index} را پیدا کن",
        "مجموعاً {count} گزینه موجود است و اندیس انتخاب {index}", "از {count} سطر، سطر شماره {index} را برگردان",
        "There are {count} choices; select choice number {index}", "From {count} entries, inspect entry {index}",
        "A collection has {count} items and target index is {index}", "Pick position {index} out of {count} records",
        "Total record count is {count}; requested record number is {index}", "The array contains {count} elements; inspect element {index}",
        "Among {count} options, return option {index}", "List size is {count}, and the desired position is {index}",
        "Use item {index} from a set containing {count} items", "There are {count} rows; show row {index}",
        "Choose the {index}th entry among {count} entries", "Dataset count {count}; selected index {index}",
        "The sequence contains {count} records but we need record {index}", "Out of {count} available objects, inspect number {index}",
    ]
    for i, t in enumerate(generic_forms): templates.append(("generic", f"generic_{i:02d}", t, ["count","index"]))

    rows: list[dict] = []
    per_template = max(4, target // len(templates))
    for family, tid, template, roles in templates:
        for _ in range(per_template):
            if family == "work": vals = distinct_values([("workers",2,80),("hours",1,24),("output",100,10000)])
            elif family == "ratio": vals = distinct_values([("total",100,9000),("ratio_a",1,12),("ratio_b",1,12)])
            elif family == "money_percent": vals = distinct_values([("money",100,200000),("percentage",5,85)])
            elif family == "distance_time": vals = distinct_values([("distance",20,1200),("hours",1,18)])
            elif family == "age": vals = distinct_values([("age",5,65),("difference",1,12),("years_later",1,20)])
            elif family == "probability": vals = distinct_values([("probability",5,90),("n",3,20),("k",1,12)]); vals["k"] = min(vals["k"], vals["n"]-1) or 1
            else: vals = distinct_values([("count",10,500),("index",1,9)])
            text = template.format(**vals)
            # 28% Persian-digit augmentation for Persian scripts.
            if re.search(r"[\u0600-\u06ff]", text) and R.random() < .28:
                text = re.sub(r"\d+", lambda m: fmt_fa(int(m.group())), text)
            rows.append({"task":"numeric_role_label","family":family,"template_id":tid,"split":template_split(tid),"text":text,"slots":vals})
    R.shuffle(rows)
    if len(rows) > target: rows = rows[:target]
    # top up while preserving template diversity
    while len(rows) < target:
        base = R.choice(rows).copy(); base["text"] = base["text"] + ("؟" if re.search(r"[\u0600-\u06ff]", base["text"]) else "?"); rows.append(base)
    return rows


def build_multistep_rows(target: int = 12000) -> list[dict]:
    specs = {
        "pct_remove_add": (
            [("percentage_remove", "percent"), ("add", "delta")],
            [
                "{initial} را {percent} درصد کم کن و بعد {delta} اضافه کن",
                "اول {percent}٪ از {initial} کم کن؛ سپس {delta} به نتیجه بیفزا",
                "از {initial} شروع کن، {percent} درصد کاهش بده و بعدش {delta} اضافه کن",
                "ابتدا مقدار {initial} را {percent} درصد کاهش بده، بعد {delta} واحد زیادش کن",
                "Starting from {initial}, remove {percent}% and then add {delta}",
                "Reduce {initial} by {percent} percent, then increase the result by {delta}",
                "Take {initial}; first subtract {percent}% of it, then add {delta}",
                "From {initial}, apply a {percent}% decrease followed by +{delta}",
            ],
        ),
        "pct_remove_subtract": (
            [("percentage_remove", "percent"), ("subtract", "delta")],
            [
                "{initial} را {percent} درصد کم کن و سپس {delta} دیگر کم کن",
                "اول از {initial}، {percent}٪ کم کن؛ بعد {delta} واحد هم کم کن",
                "از {initial} شروع کن، کاهش {percent} درصدی بده و بعد {delta} تا کم کن",
                "Reduce {initial} by {percent}% and then subtract {delta}",
                "Starting at {initial}, apply -{percent}% followed by minus {delta}",
                "Take {initial}, decrease it {percent} percent, then remove another {delta}",
                "From {initial}, remove {percent}% first and then take away {delta}",
                "Decrease {initial} by {percent} percent; afterwards remove {delta} more",
                "{initial} را اول {percent} درصد کاهش بده و بعد {delta} واحد دیگر بردار",
                "از مقدار {initial} ابتدا {percent}٪ کم کن و در مرحله دوم {delta} تا حذف کن",
            ],
        ),
        "pct_add_subtract": (
            [("percentage_add", "percent"), ("subtract", "delta")],
            [
                "{initial} را {percent} درصد زیاد کن و بعد {delta} کم کن",
                "اول {percent}٪ به {initial} اضافه کن، سپس {delta} واحد کم کن",
                "مقدار {initial}: افزایش {percent} درصدی و بعد کاهش {delta} واحدی",
                "Increase {initial} by {percent}% and then subtract {delta}",
                "Start with {initial}, add {percent} percent, then take away {delta}",
                "Apply +{percent}% to {initial}; afterwards subtract {delta}",
            ],
        ),
        "pct_add_add": (
            [("percentage_add", "percent"), ("add", "delta")],
            [
                "{initial} را {percent} درصد زیاد کن و بعد {delta} هم اضافه کن",
                "به {initial} اول {percent}٪ و بعد {delta} واحد اضافه کن",
                "از {initial} شروع کن؛ {percent} درصد رشد بده، بعد {delta} بیشتر کن",
                "Increase {initial} by {percent}% and then add {delta}",
                "Start at {initial}; grow it {percent} percent, then add {delta}",
                "Apply +{percent}% to {initial}, followed by +{delta}",
            ],
        ),
        "add_multiply": (
            [("add", "delta"), ("multiply", "factor")],
            [
                "به {initial} مقدار {delta} اضافه کن و حاصل را در {factor} ضرب کن",
                "اول {initial}+{delta} را حساب کن؛ بعد نتیجه را {factor} برابر کن",
                "از {initial} شروع کن، {delta} بیفزا و سپس ضربدر {factor}",
                "Add {delta} to {initial}, then multiply by {factor}",
                "Starting with {initial}, plus {delta}, then times {factor}",
                "Take {initial}; add {delta}; multiply the result by {factor}",
            ],
        ),
        "subtract_divide": (
            [("subtract", "delta"), ("divide", "factor")],
            [
                "از {initial} مقدار {delta} کم کن و نتیجه را بر {factor} تقسیم کن",
                "اول {delta} را از {initial} کم کن؛ بعد تقسیم بر {factor}",
                "{initial} منهای {delta} و سپس حاصل تقسیم بر {factor}",
                "Subtract {delta} from {initial}, then divide by {factor}",
                "Start with {initial}; minus {delta}; then divide the result by {factor}",
                "Take {initial}, remove {delta}, and divide what remains by {factor}",
            ],
        ),
        "discount_fee": (
            [("percentage_remove", "percent"), ("add", "delta")],
            [
                "قیمت {initial} است؛ {percent} درصد تخفیف بده و بعد {delta} هزینه ارسال اضافه کن",
                "از مبلغ {initial}، {percent}٪ تخفیف کم کن و سپس {delta} تومان هزینه بیفزا",
                "مبلغ پایه {initial}؛ تخفیف {percent} درصد؛ بعد هزینه ثابت {delta}",
                "Price is {initial}; apply {percent}% discount, then add a {delta} shipping fee",
                "Take price {initial}, reduce it by {percent} percent, then add fee {delta}",
                "Base cost {initial}; discount {percent}%; afterwards add {delta} fee",
            ],
        ),
        "inventory_sell_restock": (
            [("percentage_remove", "percent"), ("add", "delta")],
            [
                "موجودی {initial} کالا است؛ {percent} درصد فروخته می‌شود و بعد {delta} کالا اضافه می‌شود",
                "از {initial} واحد موجودی، {percent}٪ فروش برود؛ سپس {delta} واحد شارژ کن",
                "اول {percent} درصد از موجودی {initial} کم کن و بعد {delta} واحد وارد انبار کن",
                "Inventory starts at {initial}; sell {percent}% and then restock {delta} units",
                "From {initial} units, remove {percent} percent sold, then add {delta} new units",
                "Starting inventory {initial}; {percent}% sold; afterwards restock {delta}",
            ],
        ),
    }
    # Linguistic expansion changes register/instruction structure, not just numbers.
    for label,(ops,forms) in list(specs.items()):
        expanded=[]
        for f in forms:
            expanded.append(f)
            if re.search(r"[\u0600-\u06ff]",f):
                expanded += ["لطفاً مرحله‌ای حساب کن: "+f, "این عملیات را به ترتیب انجام بده: "+f, "محاسبه کن؛ "+f, f+" و نتیجه نهایی را بده"]
            else:
                expanded += ["Please calculate step by step: "+f, "Apply these operations in order: "+f, "Compute this sequence: "+f, f+" and return the final value"]
        # preserve first occurrence while deduplicating; cap at 40 templates/family
        uniq=[]
        for x in expanded:
            if x not in uniq: uniq.append(x)
        specs[label]=(ops,uniq[:40])
    rows: list[dict] = []
    tids = []
    for label, (_, forms) in specs.items():
        for i, f in enumerate(forms): tids.append((label, f"{label}_{i:02d}", f))
    each = max(5, target // len(tids))
    for label, tid, template in tids:
        for _ in range(each):
            vals = distinct_values([("initial",100,10000),("percent",5,75),("delta",5,500),("factor",2,12)])
            text = template.format(**vals)
            steps = []
            for op, argname in specs[label][0]:
                step = {"op":op}
                if op in {"percentage_remove","percentage_add"}: step["percent"] = vals["percent"]
                elif op in {"add","subtract"}: step["value"] = vals["delta"]
                elif op in {"multiply","divide"}: step["value"] = vals["factor"]
                steps.append(step)
            graph = {"initial":vals["initial"],"steps":steps}
            graph_pattern = "pct_remove_add" if label in {"discount_fee","inventory_sell_restock"} else label
            rows.append({"input":text,"pattern":graph_pattern,"scenario":label,"template_id":tid,"split":template_split(tid),"graph":graph})
    R.shuffle(rows)
    rows = rows[:target]
    while len(rows) < target:
        rows.append(R.choice(rows).copy())
    return rows


def build_code_rows(target: int = 5000) -> list[dict]:
    intents = {
        "filter_positive": [
            "تابعی بنویس که اعداد مثبت لیست را برگرداند", "یک تابع پایتون برای فیلتر اعداد بزرگ‌تر از صفر بساز",
            "از یک لیست فقط عددهای مثبت را با تابع برگردان", "کدی بنویس که مقادیر مثبت آرایه را جدا کند",
            "write a function that returns positive numbers from a list", "create Python code to filter values greater than zero",
            "implement a function that keeps only positive array elements", "make a list filter for positive numbers",
        ],
        "max_value": [
            "تابعی بساز که بیشترین عدد آرایه را پیدا کند", "بزرگ‌ترین مقدار لیست را با یک تابع پیدا کن",
            "یک تابع پایتون برای maximum لیست بنویس", "کدی میخوام که ماکزیمم آرایه رو بده",
            "write a function that finds the maximum value in an array", "create Python code to return the largest list item",
            "implement a max-value function for numbers", "make a function that returns the biggest element",
        ],
        "flask_api": [
            "یک API ساده Flask بساز", "با Flask یک endpoint ساده ایجاد کن", "کد پایتون API کوچک با flask بنویس",
            "یک سرویس Flask با مسیر health بساز", "build a simple Flask API", "create a small Flask endpoint in Python",
            "write a minimal REST API using Flask", "implement a Flask health-check route",
        ],
        "filter_even": [
            "تابعی بنویس که اعداد زوج لیست را برگرداند", "فقط عددهای زوج آرایه را فیلتر کن", "تابع پایتون برای even filter بساز",
            "کدی بساز که مقادیر زوج رو جدا کنه", "write a function to return even numbers", "filter even values from an array in Python",
            "implement an even-number list filter", "make a function that keeps numbers divisible by two",
        ],
        "sort_values": [
            "تابعی برای مرتب کردن صعودی لیست بنویس", "آرایه را از کوچک به بزرگ مرتب کن", "یک تابع پایتون برای sort کردن لیست بساز",
            "کد مرتب‌سازی صعودی مقادیر را بده", "write a function that sorts a list ascending", "create Python code to order array values from low to high",
            "implement ascending sorting for a list", "make a function that returns sorted values",
        ],
        "sum_values": [
            "تابعی بنویس که مجموع اعداد لیست را برگرداند", "جمع همه عناصر آرایه را حساب کن", "یک تابع پایتون برای sum لیست بساز",
            "کدی بده که مجموع مقادیر را حساب کند", "write a function that sums all list numbers", "create Python code to total array values",
            "implement a list sum function", "make a function that returns the total of numbers",
        ],
    }
    rows=[]
    variants=[]
    prefixes_fa=["لطفاً ","برام ","واسه من ","میخوام ",""]
    prefixes_en=["Please ","Can you ","I need you to ",""]
    for label, bases in intents.items():
        for i, base in enumerate(bases):
            prefixes = prefixes_fa if re.search(r"[\u0600-\u06ff]",base) else prefixes_en
            for j,prefix in enumerate(prefixes):
                variants.append((label,f"code_{label}_{i:02d}_{j:02d}",prefix+base))
    per=max(2,target//len(variants))
    for label,tid,text in variants:
        for _ in range(per):
            suffix = R.choice(["", " لطفاً.", " فقط خود تابع را بده.", " with a clean implementation."])
            rows.append({"input":text+suffix,"intent":label,"template_id":tid,"split":template_split(tid)})
    R.shuffle(rows); rows=rows[:target]
    while len(rows)<target: rows.append(R.choice(rows).copy())
    return rows


def build_frame_rows(numeric_rows: list[dict], multi_rows: list[dict], code_rows: list[dict], target: int = 18000) -> list[dict]:
    rows=[]
    # Positive frames from actual datasets.
    for r in numeric_rows:
        fam=r["family"]
        label={"work":"work_rate","ratio":"ratio_split","age":"age_reasoning","probability":"probability"}.get(fam)
        if label:
            rows.append({"input":r["text"],"label":label,"template_id":"frame_"+r["template_id"],"split":r["split"]})
    for r in multi_rows:
        rows.append({"input":r["input"],"label":"multistep","template_id":"frame_"+r["template_id"],"split":r["split"]})
    for r in code_rows:
        rows.append({"input":r["input"],"label":"code_generation","template_id":"frame_"+r["template_id"],"split":r["split"]})
    # Collision-heavy classes; each has many surface variants.
    system_templates=[]
    for obj in ["صدا","ولوم","volume"]:
        for verb in ["روی {x} درصد بگذار","روی {x}٪ تنظیم کن","را {x} درصد کن","set to {x} percent","مقدارش را به {x} درصد تغییر بده"]:
            system_templates.append(("system_action",f"sys_vol_{len(system_templates):02d}",f"{obj} {verb}"))
    for obj in ["روشنایی","brightness","نور صفحه"]:
        for verb in ["روی {x} درصد بگذار","روی {x}٪ تنظیم کن","را {x} درصد کن","set to {x} percent","مقدارش را به {x} درصد تغییر بده"]:
            system_templates.append(("system_action",f"sys_bri_{len(system_templates):02d}",f"{obj} {verb}"))
    sequence_templates=[]
    sequence_surface=[
        "دنباله هندسی از {a} با نسبت {r} داریم. جمله {n} چیست؟",
        "در یک تصاعد هندسی، جمله اول {a} و قدرنسبت {r} است؛ جمله {n} را پیدا کن",
        "جمله {n} دنباله هندسی با شروع {a} و نسبت {r} چند است؟",
        "دنباله‌ای هندسی با a1={a} و r={r} داریم؛ a{n} را حساب کن",
        "اگر جمله نخست دنباله هندسی {a} و نسبت مشترک {r} باشد، جمله {n} چقدر می‌شود؟",
        "تصاعد هندسی از عدد {a} شروع می‌شود و هر بار در {r} ضرب می‌شود؛ عضو {n} چیست؟",
        "در GP با جمله اولیه {a} و common ratio برابر {r}، ترم {n} را بده",
        "شروع دنباله {a} است و قدر نسبت {r}؛ مقدار جمله شماره {n}؟",
        "برای دنباله هندسی a_1={a} و q={r}، a_{n} را پیدا کن",
        "یک دنباله هندسی با پایه {a} و نسبت {r} داریم؛ جمله {n} را محاسبه کن",
        "sequence starts at {a} with ratio {r}; what is term {n}?",
        "Find term {n} of a geometric sequence whose first term is {a} and ratio is {r}",
        "A GP has a1={a} and common ratio {r}; compute term {n}",
        "For a geometric progression beginning with {a} and multiplying by {r}, find term {n}",
        "The first term is {a} and the common ratio is {r}; what is the {n}th term?",
        "Geometric series data: a1={a}, r={r}. Calculate a{n}",
        "Start a geometric sequence at {a}; each next term is times {r}. Give term {n}",
        "In a GP, initial value {a}, ratio {r}, requested index {n}; compute the value",
        "What is term number {n} when a geometric sequence has first term {a} and ratio {r}?",
        "A geometric progression begins {a} and scales by {r}; determine its {n}th member",
        "عضو شماره {n} از تصاعد هندسی با شروع {a} و ضریب {r} را حساب کن",
        "قدرنسبت {r} و جمله اول {a} است؛ جمله {n} این تصاعد هندسی چیست؟",
        "اگر دنباله هندسی با {a} آغاز شود و نسبتش {r} باشد، ترم {n} را بده",
        "دنباله GP: ابتدا {a}، نسبت {r}؛ مقدار در جایگاه {n}؟",
        "در دنباله‌ای که هندسی است، a1 برابر {a} و r برابر {r}؛ a{n} چند است؟",
        "Find a_{n} for a GP starting at {a} with multiplier {r}",
        "A geometric sequence has base term {a}, scaling factor {r}, and target position {n}; solve it",
        "Compute position {n} in the geometric progression defined by first={a}, ratio={r}",
        "Given GP({a}, ratio {r}), return the value at index {n}",
        "The geometric pattern begins with {a} and uses ratio {r}; calculate member {n}",
    ]
    for i,t in enumerate(sequence_surface): sequence_templates.append(("sequence",f"seq_{i:02d}",t))
    other_surface=[
        "امروز هوا چطوره؟","یک متن کوتاه بنویس","این جمله را ترجمه کن","فایل را باز کن","سلام چه خبر",
        "این دو گزینه را مقایسه کن","این پاراگراف را خلاصه کن","مرورگر را باز کن","الان ساعت چنده؟","بازگشت را ساده توضیح بده",
        "برای فردا برنامه بده","این خطا را دیباگ کن","یک داستان کوتاه بساز","درباره رم توضیح بده","این متن را رسمی‌تر کن",
        "compare these two options","summarize this paragraph","open the browser","what time is it","explain recursion simply",
        "plan my tasks for tomorrow","debug this error message","write a short story","explain computer memory","rewrite this sentence formally",
        "search the web for this topic","give me three creative names","translate this into English","help me make a decision","tell me a quick joke",
    ]
    other_templates=[("other",f"other_{i:02d}",t) for i,t in enumerate(other_surface)]
    for label,tid,t in system_templates:
        for _ in range(120): rows.append({"input":t.format(x=R.randint(0,100)),"label":label,"template_id":tid,"split":template_split(tid)})
    for label,tid,t in sequence_templates:
        for _ in range(220):
            vals=distinct_values([("a",2,20),("r",2,8),("n",3,10)])
            rows.append({"input":t.format(**vals),"label":label,"template_id":tid,"split":template_split(tid)})
    for label,tid,t in other_templates:
        for _ in range(100): rows.append({"input":t,"label":label,"template_id":tid,"split":template_split(tid)})
    R.shuffle(rows)
    if len(rows)>target: rows=rows[:target]
    return rows


def sparse_matrix(samples, feature_fn, size):
    indptr=[0]; indices=[]; values=[]
    for sample in samples:
        idx,val=feature_fn(sample,size); order=np.argsort(idx); idx=idx[order]; val=val[order]
        indices.extend(int(x) for x in idx); values.extend(float(x) for x in val); indptr.append(len(indices))
    return csr_matrix((np.asarray(values,np.float32),np.asarray(indices,np.int32),np.asarray(indptr,np.int32)),shape=(len(samples),size),dtype=np.float32)


def classification_metrics(clf,x,y,labels):
    pred=clf.predict(x); cm=confusion_matrix(y,pred,labels=np.arange(len(labels)))
    pc={}
    for i,l in enumerate(labels):
        den=int(cm[i].sum()); pc[l]=float(cm[i,i]/den) if den else None
    return {"examples":len(y),"accuracy":float(accuracy_score(y,pred)),"per_class_accuracy":pc,"confusion_matrix":cm.tolist()}


def fit_save(rows, labels_key, feature_builder, model_cls, model_path: Path, size: int):
    splitrows={s:[r for r in rows if r["split"]==s] for s in ("train","validation","test")}
    labels=tuple(sorted({r[labels_key] for r in rows})); lid={l:i for i,l in enumerate(labels)}
    xs={s:sparse_matrix(arr,lambda r,z:feature_builder(r,z),size) for s,arr in splitrows.items()}
    ys={s:np.asarray([lid[r[labels_key]] for r in arr],np.int64) for s,arr in splitrows.items()}
    clf=SGDClassifier(loss="log_loss",penalty="l2",alpha=3e-6,max_iter=50,tol=1e-5,random_state=SEED,class_weight="balanced",average=False).fit(xs["train"],ys["train"])
    rep={s:classification_metrics(clf,xs[s],ys[s],labels) for s in xs}
    weights=np.asarray(clf.coef_,np.float32); bias=np.asarray(clf.intercept_,np.float32)
    np.savez_compressed(model_path,format=np.asarray(model_cls.FORMAT),labels=np.asarray(labels),weights=weights,bias=bias,
        parameter_count=np.asarray(weights.size+bias.size,np.int64),training_examples=np.asarray(len(splitrows["train"]),np.int64),dataset_version=np.asarray(VERSION),holdout_strategy=np.asarray("template_id_disjoint"))
    return {"labels":labels,"parameters":int(weights.size+bias.size),"splits":{s:len(v) for s,v in splitrows.items()},"metrics":rep}


def number_samples(rows):
    out=[]
    for r in rows:
        text=r["text"]
        ascii_text=text.translate(DIGITS)
        byval=defaultdict(list)
        for role,val in r["slots"].items(): byval[int(val)].append(role)
        for m in re.finditer(r"(?<![\w.])\d+(?:[.,]\d+)?",ascii_text):
            val=int(float(m.group().replace(",","."))); roles=byval.get(val,[])
            label=roles[0] if len(roles)==1 else "other"
            out.append({"text":ascii_text,"start":m.start(),"end":m.end(),"label":label,"split":r["split"],"template_id":r["template_id"]})
    return out


def main():
    numeric=build_numeric_rows(24000)
    multi=build_multistep_rows(12000)
    code=build_code_rows(5000)
    frame=build_frame_rows(numeric,multi,code,18000)
    assign_stratified_splits(numeric,"family")
    assign_stratified_splits(multi,"pattern")
    assign_stratified_splits(code,"intent")
    assign_stratified_splits(frame,"label")
    write_jsonl(ROOT/"datasets"/"numeric_roles_v19"/"numeric_role_24000_real.jsonl",numeric)
    write_jsonl(ROOT/"datasets"/"multistep_v19"/"execution_graph_12000_real.jsonl",multi)
    write_jsonl(ROOT/"datasets"/"code_v19"/"code_intelligence_5000_real.jsonl",code)
    write_jsonl(ROOT/"datasets"/"semantic_ir_v19"/"semantic_frame_18000_real.jsonl",frame)

    nums=number_samples(numeric)
    num_report=fit_save(nums,"label",lambda r,z:NumericRoleTaggerV19.sparse_features(r["text"],r["start"],r["end"],z),NumericRoleTaggerV19,ROOT/"models"/"numeric_role_v19.npz",NUM_SIZE)
    frame_report=fit_save(frame,"label",lambda r,z:SemanticFrameClassifierV19.sparse_features(r["input"],z),SemanticFrameClassifierV19,ROOT/"models"/"semantic_frame_v19.npz",FRAME_SIZE)
    exec_report=fit_save(multi,"pattern",lambda r,z:ExecutionPatternClassifierV19.sparse_features(r["input"],z),ExecutionPatternClassifierV19,ROOT/"models"/"execution_pattern_v19.npz",EXEC_SIZE)
    code_report=fit_save(code,"intent",lambda r,z:CodeIntentClassifierV19.sparse_features(r["input"],z),CodeIntentClassifierV19,ROOT/"models"/"code_intent_v19.npz",CODE_SIZE)

    def diversity(rows, family_key=None):
        if family_key:
            d=defaultdict(set)
            for r in rows:d[r[family_key]].add(r["template_id"])
            return {k:len(v) for k,v in sorted(d.items())}
        return len({r["template_id"] for r in rows})
    report={
        "format":"jarvis-v19-real-training-report","dataset_version":VERSION,"seed":SEED,"holdout_strategy":"template_id_disjoint",
        "datasets":{
            "numeric_roles":{"rows":len(numeric),"templates":diversity(numeric,"family")},
            "multistep":{"rows":len(multi),"templates":diversity(multi,"pattern")},
            "code":{"rows":len(code),"templates":diversity(code,"intent")},
            "semantic_frame":{"rows":len(frame),"templates":diversity(frame,"label")},
        },
        "models":{"numeric_role_v19":num_report,"semantic_frame_v19":frame_report,"execution_pattern_v19":exec_report,"code_intent_v19":code_report},
    }
    # Quality gates use template-disjoint test sets, not random same-template samples.
    test_accs={k:v["metrics"]["test"]["accuracy"] for k,v in report["models"].items()}
    report["quality_gates"]={"minimum":{"numeric_role_v19":.86,"semantic_frame_v19":.90,"execution_pattern_v19":.90,"code_intent_v19":.92},"observed":test_accs}
    report["passed"]=all(test_accs[k]>=report["quality_gates"]["minimum"][k] for k in test_accs)
    out=ROOT/"reports"/"v19_real_training_report.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report["passed"]: raise SystemExit(2)

if __name__=="__main__": main()
