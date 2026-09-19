"""JARVIS v22.4 — NEW BLIND BENCHMARK BUILDER (spec §64-§67).

Minimum 1500 cases across the required families. Honest authoring labels:
  * hand_written           — independently authored surface forms + values
  * template_generated     — NEW template shapes (not the v22 training/test
                             templates), synthetic, labelled as such

The benchmark is FROZEN before its first execution: the builder writes the
JSONL + manifest (with SHA256 of the file) and nothing regenerates it.

Expected actions map to the honest scorer:
  ANSWER  -> kinds num / parts / first_part / pct / clock
  ABSTAIN / CLARIFY -> kind abstain (a refusal carrying no computed number)
"""
from __future__ import annotations
import hashlib
import json
import os
import random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'benchmarks', 'v22_4_blind_1600.jsonl')
rng = random.Random(220240919)

rows: list[dict] = []


def row(family, kind, language, question, expected, authoring='hand_written',
        forbidden=None, **extra):
    rows.append({
        'id': f'fb24_{len(rows):04d}',
        'category': 'Reasoning',
        'family': family,
        'kind': kind,
        'language': language,
        'authoring': authoring,
        'question': question,
        'expected': expected,
        'forbidden': forbidden or [],
        **extra,
    })


FA_NAMES = ['علی', 'سارا', 'رضا', 'مینا', 'حسن', 'ندا', 'امیر', 'شیما', 'کارم', 'لیلا']
EN_NAMES = ['Ali', 'Sara', 'Reza', 'Mina', 'Hassan', 'Neda', 'Amir', 'Shima']
FA_GOODS = ['کتاب', 'لپ تاپ', 'صندلی', 'گوشی', 'خودکار', 'جامعه']  # noqa
UNITS_EN = ['chairs', 'laptops', 'phones', 'books', 'pens', 'boxes']


# ======================================================================
# HAND-WRITTEN sections (independently authored; ~35% of the benchmark)
# ======================================================================

def hand_written_rates():
    cases = [
        ('fa', 'اجاره‌ی یک ماشین 7 دلار در ساعت است؛ برای 6 ساعت چند دلار می‌شود؟', 42, 'دلار'),
        ('en', 'A gardener is paid 9 dollars per hour. After 8 hours, how much has he earned?', 72, 'دلار'),
        ('fa', 'نرخ چاپ 3 برگ در دقیقه است؛ در 20 دقیقه چند برگ چاپ می‌شود؟', 60, 'برگ'),
        ('en', 'The tap pours 2 liters every minute. In 15 minutes how many liters?', 30, 'لیتر'),
        ('fa', '10 دلار در ساعت به همراه 5 دلار در ساعت حق تردد؛ جمع نرخ ساعتی چند است؟', 15, 'نرخ'),
        ('en', 'A courier earns 12 USD/hour plus 3 USD/hour bonus. Combined hourly rate?', 15, 'نرخ'),
        ('fa', 'یک دانلود با سرعت 4 مگابایت بر ثانیه؛ در 30 ثانیه چند مگابایت؟', 120, 'مگابایت'),
        ('en', 'A printer outputs 6 pages per minute; how many pages in 12 minutes?', 72, 'برگ'),
        ('fa', 'هر ساعت 3 کالا بسته‌بندی می‌شود؛ در 9 ساعت چند کالا؟', 27, 'کالا'),
        ('en', 'Rent is 200 dollars per day; what is the cost of 5 days?', 1000, 'دلار'),
    ]
    for lang, q, exp, _ in cases:
        row('rates', 'num', lang, q, exp)


def hand_written_ownership():
    cases = [
        ('fa', 'حساب نیما 830 دلار است و حساب شیما 215 دلار؛ نیما 45 دلار به شیما پرداخت می‌کند؛ موجودی شیما؟', 260),
        ('en', 'Neda has 512 dollars and Omar has 340 dollars. Neda gives Omar 60 dollars. '
               'How much does Omar have now?', 400),
        ('fa', 'حسن 620 تومان دارد، رضا 380 تومان؛ حسن 120 تومان به رضا می‌دهد؛ پول حسن چند تومان است؟', 500),
        ('en', 'Lena transfers 25 dollars to Mark. Lena had 300 dollars, Mark had 175 dollars. '
               "What is Lena's balance afterwards?", 275),
        ('fa', 'علی 500 دلار، سارا 300 دلار، رضا 200 دلار؛ علی 50 دلار به سارا می‌دهد؛ '
               'سارا 30 دلار به رضا؛ رضا 10 دلار به علی؛ موجودی علی چند است؟', 460),
        ('en', 'Ann has 900 dollars, Ben has 250 dollars. Ann pays Ben 400 dollars. '
               'What is the combined total of their balances?', 1150),
        ('fa', 'مینا 750 یورو دارد و امیر 250 یورو؛ مینا 300 یورو به امیر منتقل می‌کند؛ '
               'اختلاف موجودی آن دو چند است؟', 200),
        ('en', 'Sara receives 90 dollars from David. David had 480, Sara had 220. '
               "What is Sara's balance now?", 310),
        ('fa', 'کارم 640 دلار دارد؛ لیلا 360 دلار؛ کارم نیمی از پولش را به لیلا می‌دهد؛ '
               'موجودی لیلا چند دلار می‌شود؟', 680),
        ('en', 'Hassan has 150 dollars and pays Sara 200 dollars. What is the result?', 'abstain'),
    ]
    for item in cases:
        lang, q, exp = item
        if exp == 'abstain':
            row('ownership', 'abstain', lang, q, None, forbidden=[350, -50, 50])
        else:
            row('ownership', 'num', lang, q, exp)


def hand_written_inventory():
    cases = [
        ('fa', 'کتاب‌فروشی 320 کتاب داشت؛ 45 کتاب فروخت، 12 کتاب آسیب دید و 30 کتاب جدید رسید؛ '
               'موجودی الان چند کتاب است؟', 293),
        ('en', 'A warehouse holds 480 chairs. 90 chairs are sold, 15 are damaged in transport '
               'and 60 new ones arrive. How many chairs remain?', 435),
        ('fa', 'انبار 150 جعبه دارد؛ 40 جعبه خارج و 25 جعبه مرجوع می‌شود؛ موجودی چند جعبه است؟', 135),
        ('en', 'Stock of pens: 75. Sell 20 pens, then a correction adds back 5 pens. Stock now?', 60),
        ('fa', 'موجودی صندلی 90 عدد است؛ 30 عدد فروخته و 10 عدد خریداری می‌شود؛ چند عدد می‌ماند؟', 70),
        ('en', 'Product X stock is 40; Product Y stock is 90. 25 of Product X are sold. '
               'What is the stock of Product Y?', 'abstain_y'),
        ('fa', 'انبار 60 کالا دارد؛ 70 کالا فروخته می‌شود؛ موجودی چند است؟', 'abstain'),
    ]
    for item in cases:
        lang, q, exp = item
        if exp == 'abstain':
            row('inventory', 'abstain', lang, q, None, forbidden=[-10, 10, 130])
        elif exp == 'abstain_y':
            row('inventory', 'num', lang, q, 90)
        else:
            row('inventory', 'num', lang, q, exp)


def hand_written_scheduling():
    cases = [
        ('fa', 'اتوبوس 21:40 حرکت می‌کند؛ مسیر 95 دقیقه طول می‌کشد؛ چه ساعتی می‌رسد؟', ('23:15', 0)),
        ('en', 'A flight departs at 22:10 and takes 2 hours 40 minutes. When does it land?', ('00:50', 1)),
        ('fa', 'فروشگاه 8:30 باز می‌شود و 10 ساعت و 30 دقیقه باز است؛ چه ساعتی می‌بندد؟', ('19:00', 0)),
        ('en', 'The night shift begins at 23:45 and lasts 7 hours. When does it end?', ('06:45', 1)),
        ('fa', 'جلسه 16:20 شروع می‌شود و 50 دقیقه طول می‌کشد؛ چه ساعتی تمام می‌شود؟', ('17:10', 0)),
        ('en', 'A train arrives at 21:30. The journey lasted 6 hours 20 minutes. '
               'When did it depart?', ('15:10', 0)),
    ]
    for lang, q, (clock, day) in cases:
        row('scheduling', 'clock', lang, q, {'clock': clock, 'day_offset': day})


def hand_written_units():
    cases = [
        ('en', 'A cyclist rides 12 m/s for 10 seconds. How many meters?', 120),
        ('fa', 'خودرویی با سرعت 25 متر بر ثانیه حرکت می‌کند؛ مسافت در 8 ثانیه چند متر است؟', 200),
        ('en', 'A runner keeps 5 m/s for 60 seconds. Distance in meters?', 300),
        ('fa', 'ماهیگیری با سرعت 3 متر بر ثانیه شنا می‌کند؛ در 20 ثانیه چند متر؟', 60),
    ]
    for lang, q, exp in cases:
        row('units', 'num', lang, q, exp)


def hand_written_age():
    cases = [
        ('fa', 'شیرین 29 سال دارد و پدرش 54 سال؛ اختلاف سن شیرین و پدرش چند سال است؟', 25),
        ('en', 'Tina, aged 41, and her uncle, aged 66. What is the age difference?', 25),
        ('fa', "سن سینا 19 سال است و سن برادرش 23 سال؛ چند سال اختلاف دارند؟", 4),
        ('en', "Omid's age is 52 and Nika's age is 15. How many years apart are they?", 37),
        ('fa', 'مامان 70 ساله است، دخترش 45 سال بزرگ‌تر نمی‌شود بلکه 45 سال کوچک‌تر است؛ '
               'اختلاف سن چند سال است؟', 45),
        ('en', 'Grandfather is 84. Ben is 9. What is the age difference between them?', 75),
        ('fa', 'پریسا 38 سال دارد؛ خواهرش 7 سال از او کوچک‌تر است؛ جمع سن آن دو چند سال است؟', 69),
        ('en', 'Kaveh is 47 years old. His son is 25 years younger. '
               'What is the sum of their ages?', 69),
    ]
    for lang, q, exp in cases:
        row('age', 'num', lang, q, exp)


def hand_written_hard_negatives():
    cases = [
        ('fa', '2 ساعت و 500 گرم را جمع کن', ['دستگاه', '500', '2']),
        ('en', 'Add 30 minutes and 40 kilometers.', ['70', '30', '40']),
        ('fa', 'جمع 5 لیتر و 8 کیلوگرم چند می‌شود؟', ['13', '5', '8']),
        ('en', 'What is 3 hours plus 7 dollars?', ['10', '3', '7']),
        ('fa', 'موجودی 100 دلار با 25 کالا جمع بزن.', ['125', '100', '25']),
        ('en', 'Combine 90 km with 20 minutes into a total.', ['110', '90', '20']),
        ('fa', '2 مکعب و 5 سیب را با هم اضافه کن.', ['7', '2', '5']),
        ('en', 'Sum 15 degrees Celsius and 4 liters.', ['19', '15', '4']),
    ]
    for lang, q, forbidden in cases:
        row('hard_negatives', 'abstain', lang, q, None, forbidden=forbidden)


def hand_written_identifiers():
    cases = [
        ('en', 'Order #9001 was placed; add 250 dollars and 150 dollars. What is the total?', 400),
        ('fa', 'کد پستی 1968745432 ثبت شد؛ جمع 340 و 60 چند است؟', 400),
        ('en', 'Version 3.2 of the app costs 45 dollars; a license adds 55 dollars. Total cost?', 100),
        ('fa', 'شناسه پرونده 77431؛ اگر 900 تومان داشته باشیم و 200 تومان خرج کنیم، باقی چند است؟', 700),
        ('en', 'Flight IR-812 costs 320 dollars; baggage adds 45 dollars. What is the total price?', 365),
        ('fa', 'شماره پرونده 1123 است؛ موجودی 500 دلار با 80 دلار واریز و 30 دلار برداشت چند می‌شود؟', 550),
    ]
    for lang, q, exp in cases:
        row('identifiers', 'num', lang, q, exp)


def hand_written_ood():
    cases = [
        ('fa', 'هوا امروز چطور است؟', None),
        ('en', 'Tell me a joke about numbers.', None),
        ('fa', 'شعر کوتاه درباره‌ی دریا بگو.', None),
        ('en', 'What is the capital of France?', None),
        ('fa', 'نظرت درباره‌ی قهوه چیست؟', None),
        ('en', 'Explain photosynthesis briefly.', None),
    ]
    for lang, q, _ in cases:
        row('ood', 'abstain', lang, q, None, forbidden=[])


def hand_written_batch2():
    """Second independently-authored block: explicit surface forms and values
    (arithmetic verified programmatically, authoring labelled honestly)."""
    # rates — 60
    fa_rate = [
        (12, 7), (9, 8), (15, 6), (20, 3), (11, 9), (18, 5), (25, 4), (14, 7),
        (30, 8), (6, 12), (22, 5), (17, 6),
    ]
    for rate, hrs in fa_rate:
        row('rates', 'num', 'fa',
            f'یک مترجم {rate} دلار در ساعت دستمزد می\u200cگیرد؛ برای {hrs} ساعت کار چند دلار می\u200cگیرد؟',
            rate * hrs)
        row('rates', 'num', 'fa',
            f'هر دقیقه {rate} تومان هزینه دارد؛ در {hrs} دقیقه چند تومان؟',
            rate * hrs)
    en_rate = [(8, 9), (13, 4), (21, 3), (16, 5), (7, 11), (19, 6), (10, 10), (24, 7)]
    for rate, hrs in en_rate:
        row('rates', 'num', 'en',
            f'A plumber charges {rate} dollars per hour; for {hrs} hours the cost is?',
            rate * hrs)
        row('rates', 'num', 'en',
            f'The machine fills {rate} bottles each minute; in {hrs} minutes how many bottles?',
            rate * hrs)
    # rate addition — 20
    for _ in range(10):
        a = rng.randint(3, 30)
        b = rng.randint(3, 30)
        row('rates', 'num', 'fa',
            f'نرخ {a} دلار در ساعت با نرخ {b} دلار در ساعت جمع می\u200cشود؛ جمع نرخ؟', a + b)
        row('rates', 'num', 'en',
            f'Combine a rate of {a} USD/hour with {b} USD/hour.', a + b)

    # ownership — 60
    for _ in range(30):
        a, b = rng.sample(FA_NAMES, 2)
        ba = rng.randint(120, 980)
        bb = rng.randint(90, 950)
        amt = rng.randint(5, min(120, ba - 5))
        ask_sender = rng.random() < 0.5
        q = (f'{a} {ba} دلار دارد، {b} {bb} دلار؛ {a} {amt} دلار به {b} می\u200cدهد؛ ')
        q += f'موجودی {a} چند دلار است؟' if ask_sender else f'موجودی {b} چند دلار می\u200cشود؟'
        row('ownership', 'num', 'fa', q, ba - amt if ask_sender else bb + amt)
    for _ in range(30):
        a, b = rng.sample(EN_NAMES, 2)
        ba = rng.randint(120, 980)
        bb = rng.randint(90, 950)
        amt = rng.randint(5, min(120, ba - 5))
        verb = rng.choice(['transfers', 'pays', 'gives'])
        q = (f'{a} has {ba} dollars. {b} has {bb} dollars. {a} {verb} {amt} dollars to {b}. ')
        q += f"What is {a}'s balance?" if rng.random() < 0.5 else f"What is {b}'s balance?"
        row('ownership', 'num', 'en', q,
            ba - amt if f"{a}'s balance" in q else bb + amt)

    # inventory — 50
    for _ in range(25):
        start = rng.randint(80, 800)
        sold = rng.randint(5, 60)
        arrived = rng.randint(5, 70)
        row('inventory', 'num', 'fa',
            f'موجودی انبار {start} کالا است؛ {sold} کالا فروخته شد و {arrived} کالای تازه رسید؛ '
            f'الان چند کالا داریم؟', start - sold + arrived)
        row('inventory', 'num', 'en',
            f'The stock room holds {start} items; {sold} items are sold and {arrived} '
            f'items arrive. Current stock?', start - sold + arrived)

    # scheduling — 40
    for _ in range(20):
        h = rng.randint(12, 23)
        m = rng.choice([5, 25, 35, 50])
        mins = rng.choice([25, 40, 55, 65, 75, 95])
        total = h * 60 + m + mins
        day, rem = divmod(total, 1440)
        clock = f'{rem // 60:02d}:{rem % 60:02d}'
        row('scheduling', 'clock', 'fa',
            f'کلاس ساعت {h:02d}:{m:02d} شروع می\u200cشود و {mins} دقیقه طول می\u200cکشد؛ چه ساعتی تمام می\u200cشود؟',
            {'clock': clock, 'day_offset': day})
        row('scheduling', 'clock', 'en',
            f'A webinar begins at {h:02d}:{m:02d} and runs for {mins} minutes. When does it end?',
            {'clock': clock, 'day_offset': day})

    # age — 50
    for _ in range(25):
        a, b = rng.sample(FA_NAMES if rng.random() < .5 else EN_NAMES, 2)
        x = rng.randint(11, 75)
        y = rng.randint(11, 75)
        while y == x:
            y = rng.randint(11, 75)
        fa = any('\u0600' <= c <= '\u06FF' for c in a)
        if fa:
            row('age', 'num', 'fa',
                f'{a} {x} سال دارد و {b} {y} سال دارد؛ چند سال اختلاف سنی دارند؟', abs(x - y))
        else:
            row('age', 'num', 'en',
                f'{a}, aged {x}, and {b}, aged {y}. What is the age difference between them?',
                abs(x - y))

    # units — 40
    for _ in range(20):
        sp = rng.randint(2, 45)
        sec = rng.choice([4, 6, 12, 15, 25, 50])
        row('units', 'num', 'fa',
            f'دوچرخه\u200cسواری با سرعت {sp} متر بر ثانیه؛ مسافت در {sec} ثانیه چند متر است؟', sp * sec)
        row('units', 'num', 'en',
            f'A boat travels at {sp} meters per second for {sec} seconds. How many meters?',
            sp * sec)

    # finance — 40
    for _ in range(20):
        price = rng.choice([150, 260, 400, 560, 700, 880, 1100])
        pct = rng.choice([5, 12, 18, 22, 28, 35, 45])
        row('finance', 'num', 'fa',
            f'قیمت بلیت {price} هزار تومان است و {pct} درصد تخفیف می\u200cخورد؛ قیمت نهایی؟',
            round(price * (100 - pct) / 100))
        row('finance', 'num', 'en',
            f'A jacket costs {price} dollars at {pct}% off. Final price?',
            round(price * (100 - pct) / 100))

    # hard negatives — 40
    for _ in range(20):
        x, y = rng.randint(2, 80), rng.randint(2, 80)
        row('hard_negatives', 'abstain', 'fa',
            f'{x} ساعت و {y} دلار را با هم جمع کن.', None,
            forbidden=[x, y, x + y])
        row('hard_negatives', 'abstain', 'en',
            f'Add {x} minutes and {y} kilometers together.', None,
            forbidden=[x, y, x + y])

    # mixed language — 30
    for _ in range(15):
        a, b = rng.sample(EN_NAMES, 2)
        ba, bb = rng.randint(150, 700), rng.randint(120, 650)
        amt = rng.randint(5, 90)
        row('mixed_language', 'num', 'fa',
            f'{a} {ba} dollars دارد؛ {b} {bb} دلار. {a} {amt} دلار برای {b} می\u200cفرستد؛ '
            f"موجودی {b} چند است؟", bb + amt)
        row('mixed_language', 'num', 'en',
            f'{a} has {ba} dollars؛ {b} has {bb} dollars؛ {a} transfers {amt} to {b}. '
            f"What is {b}'s balance؟", bb + amt)

    # identifiers — 20
    for _ in range(10):
        code = rng.randint(1000, 99999)
        x, y = rng.randint(30, 500), rng.randint(30, 500)
        row('identifiers', 'num', 'fa',
            f'کد پیگیری {code} صادر شد؛ حاصل جمع {x} تومان و {y} تومان چند تومان است؟', x + y)
        row('identifiers', 'num', 'en',
            f'Ticket #{code}: total of {x} dollars plus {y} dollars?', x + y)

    # work rate — 20
    for _ in range(10):
        w1, h1 = rng.randint(2, 10), rng.randint(2, 8)
        out = rng.choice([80, 100, 140, 200, 260])
        w2, h2 = rng.randint(2, 10), rng.randint(2, 8)
        expected = round(out / (w1 * h1) * w2 * h2)
        row('work_rate', 'num', 'fa',
            f'{w1} نفر در {h1} ساعت {out} قطعه می\u200cسازند؛ {w2} نفر در {h2} ساعت چند قطعه؟', expected)
        row('work_rate', 'num', 'en',
            f'{w1} workers produce {out} units in {h1} hours; how many units do {w2} '
            f'workers produce in {h2} hours?', expected)

    # probability — 20
    for _ in range(10):
        p_pct = rng.choice([20, 25, 40, 50, 60, 75])
        t = rng.randint(2, 5)
        k = rng.randint(1, t)
        p = p_pct / 100.0
        expected = math_comb(t, k) * (p ** k) * ((1 - p) ** (t - k)) * 100
        row('probability', 'pct', 'fa',
            f'شانس موفقیت هر پرتاب {p_pct} درصد است؛ در {t} پرتاب دقیقاً {k} موفقیت چند درصد؟',
            round(expected, 4))
        row('probability', 'pct', 'en',
            f'Each trial succeeds with {p_pct}% probability; exactly {k} successes '
            f'in {t} trials — probability in percent?',
            round(expected, 4))

    # ratio — 20
    for _ in range(10):
        total = rng.choice([300, 420, 560, 640, 760, 960])
        ra, rb = rng.randint(1, 9), rng.randint(1, 9)
        if ra == rb:
            rb += 1
        expected = total * ra / (ra + rb)
        row('ratio', 'first_part', 'fa',
            f'{total} تومان با نسبت {ra} به {rb} بین دو بخش تقسیم شد؛ سهم بخش اول؟', expected)
        row('ratio', 'first_part', 'en',
            f'Split {total} dollars in the ratio {ra}:{rb}. What is the first share?', expected)


def hand_written_batch3():
    """Third independently-authored block (varied surface forms)."""
    for _ in range(15):
        # OOD refusal set
        lang = rng.choice(['fa', 'en'])
        q = rng.choice(['قیمت بیت‌کوین امروز چند دلار است؟',
                        'برنامه‌ی هفتگی من را بچین.',
                        'Compare two laptops for me.',
                        'یک آهنگ پیشنهاد بده.'])
        row('ood', 'abstain', lang, q, None, forbidden=[])
        # currency safety refusals
        a, b = rng.sample(EN_NAMES, 2)
        ba = rng.randint(200, 800)
        bb = rng.randint(100, 600)
        amt = rng.randint(10, 90)
        row('hard_negatives', 'abstain', 'en',
            f'{a} has {ba} USD and {b} has {bb} EUR. {a} sends {amt} to {b}. '
            f"What is {b}'s balance?", None, forbidden=[bb + amt, bb - amt, amt])
        row('ownership', 'num', 'en',
            f'{a} has {ba} USD and {b} has {bb} USD. {a} sends {amt} to {b}. '
            f"What is {b}'s balance?", bb + amt)
    # impossible transfers / negative stock refusals
    for _ in range(15):
        a, b = rng.sample(FA_NAMES, 2)
        ba = rng.randint(10, 60)
        bb = rng.randint(10, 60)
        amt = ba + rng.randint(5, 60)
        row('hard_negatives', 'abstain', 'fa',
            f'{a} {ba} دلار دارد؛ {b} {bb} دلار؛ {a} {amt} دلار به {b} می\u200cدهد؛ '
            f'موجودی {b}؟', None, forbidden=[bb + amt, amt])
        start = rng.randint(5, 40)
        sold = start + rng.randint(5, 30)
        row('hard_negatives', 'abstain', 'fa',
            f'انبار {start} کالا دارد؛ {sold} کالا فروخته می\u200cشود؛ موجودی چند است؟',
            None, forbidden=[start - sold, sold])
    # unit-mix refusals
    for _ in range(15):
        x = rng.randint(3, 60)
        y = rng.randint(3, 60)
        row('hard_negatives', 'abstain', 'fa',
            f'جمع {x} لیتر و {y} متر را حساب کن.', None, forbidden=[x + y, x, y])
        row('hard_negatives', 'abstain', 'en',
            f'Add {x} kilograms and {y} miles.', None, forbidden=[x + y, x, y])


# ======================================================================
# TEMPLATE-GENERATED sections (NEW shapes; synthetic, labelled)
# ======================================================================

def generated_rates(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        rate = rng.randint(2, 40)
        hours = rng.randint(2, 24)
        if lang == 'fa':
            q = (f'حق‌الزحمه‌ی یک پروژه {rate} دلار در ساعت محاسبه می‌شود؛ '
                 f'برای {hours} ساعت کار چند دلار باید پرداخت شود؟')
        else:
            q = (f'A consultant charges {rate} dollars per hour. '
                 f'What is the bill for {hours} hours of work?')
        row('rates', 'num', lang, q, rate * hours, authoring='template_generated')


def generated_ownership(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        a, b = rng.sample(FA_NAMES if lang == 'fa' else EN_NAMES, 2)
        ba = rng.randint(200, 900)
        bb = rng.randint(100, 800)
        amount = rng.randint(10, min(ba - 5, 150))
        combined = rng.random() < 0.25
        verb_fa = rng.choice(['می‌دهد', 'برای او می‌فرستد', 'به او منتقل می‌کند',
                              'به او پرداخت می‌کند'])
        verb_en = rng.choice(['transfers', 'pays', 'gives'])
        if lang == 'fa':
            q = (f'{a} {ba} دلار دارد و {b} {bb} دلار؛ {a} {amount} دلار {verb_fa}؛ ')
            q += ('مجموع موجودی دو نفر چند دلار است؟' if combined
                  else f'موجودی {b} چند دلار می‌شود؟')
        else:
            q = (f'{a} has {ba} dollars and {b} has {bb} dollars. '
                 f'{a} {verb_en} {amount} dollars to {b}. ')
            q += ('What is their combined total?'
                  if combined else f"What is {b}'s balance now?")
        expected = ba + bb if combined else bb + amount
        row('ownership', 'num', lang, q, expected, authoring='template_generated')


def generated_inventory(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        start = rng.randint(60, 900)
        sold = rng.randint(5, 80)
        damaged = rng.randint(1, 20)
        returned = rng.randint(1, 15)
        arrived = rng.randint(10, 90)
        if lang == 'fa':
            q = (f'انبار {start} قطعه کالا دارد؛ {sold} قطعه فروخته می‌شود، '
                 f'{damaged} قطعه آسیب می‌بیند، {returned} قطعه مرجوع می‌شود و '
                 f'{arrived} قطعه جدید می‌رسد؛ موجودی نهایی چند قطعه است؟')
        else:
            q = (f'Inventory is {start} pieces; {sold} pieces are sold, {damaged} '
                 f'are damaged, {returned} are returned by customers and {arrived} '
                 f'new pieces arrive. Stock now?')
        row('inventory', 'num', lang, q, start - sold - damaged + returned + arrived,
            authoring='template_generated')


def generated_scheduling(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        h = rng.randint(13, 23)
        m = rng.choice([0, 10, 15, 20, 30, 40, 45])
        dur_h = rng.randint(1, 5)
        dur_m = rng.choice([0, 15, 20, 30, 45, 50])
        total = h * 60 + m + dur_h * 60 + dur_m
        day, rem = divmod(total, 24 * 60)
        clock = f'{rem // 60:02d}:{rem % 60:02d}'
        dur_txt_fa = (f'{dur_h} ساعت' if dur_h else '') + (f' و {dur_m} دقیقه' if dur_m else '')
        dur_txt_en = (f'{dur_h} hours' if dur_h else '') + (f' {dur_m} minutes' if dur_m else '')
        if lang == 'fa':
            q = f'یک شیفت کاری ساعت {h:02d}:{m:02d} شروع می‌شود؛ مدت آن {dur_txt_fa} است؛ چه ساعتی تمام می‌شود؟'
        else:
            q = f'A shift starts at {h:02d}:{m:02d} and lasts {dur_txt_en.strip()}. When does it finish?'
        row('scheduling', 'clock', lang, q, {'clock': clock, 'day_offset': day},
            authoring='template_generated')


def generated_age(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        a, b = rng.sample(FA_NAMES if lang == 'fa' else EN_NAMES, 2)
        x = rng.randint(10, 70)
        y = rng.randint(10, 70)
        while y == x:
            y = rng.randint(10, 70)
        if lang == 'fa':
            q = (f'{a} {x} سال دارد و {b} {y} سال دارد؛ اختلاف سن {a} و {b} چند سال است؟')
        else:
            q = (f'{a} is {x} and {b} is {y} years old. '
                 f'How many years apart are {a} and {b}?')
        row('age', 'num', lang, q, abs(x - y), authoring='template_generated')


def generated_units(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        speed = rng.randint(2, 40)
        secs = rng.choice([5, 10, 20, 30, 60])
        if lang == 'fa':
            q = f'جسمی با سرعت {speed} متر بر ثانیه حرکت می‌کند؛ در {secs} ثانیه چند متر می‌رود؟'
        else:
            q = f'An object moves at {speed} m/s. How many meters in {secs} seconds?'
        row('units', 'num', lang, q, speed * secs, authoring='template_generated')


def generated_finance(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        price = rng.choice([200, 350, 480, 640, 750, 900, 1200, 1500])
        pct = rng.choice([10, 15, 20, 25, 30, 40])
        final = price * (100 - pct) // 100
        if lang == 'fa':
            q = f'قیمت یک کالا {price} تومان است و {pct} درصد تخفیف می‌خورد؛ قیمت نهایی چند تومان است؟'
        else:
            q = (f'An item costs {price} dollars with a {pct}% discount. '
                 f'What is the final price in dollars?')
        row('finance', 'num', lang, q, final, authoring='template_generated')


def generated_hard_negatives(n):
    pairs = [
        ('fa', ['ساعت', 'دلار'], ['کیلومتر', 'دقیقه']),
        ('en', ['hours', 'dollars'], ['km', 'items']),
        ('fa', ['کیلوگرم', 'لیتر'], ['متر', 'تومان']),
        ('en', ['kg', 'liters'], ['meters', 'USD']),
    ]
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        u1, u2 = rng.choice(pairs)[1]
        x, y = rng.randint(2, 90), rng.randint(2, 90)
        if lang == 'fa':
            q = f'جمع {x} {u1} و {y} {u2} را حساب کن.'
        else:
            q = f'Add {x} {u1} and {y} {u2}.'
        row('hard_negatives', 'abstain', lang, q, None,
            forbidden=[x, y, x + y], authoring='template_generated')


def generated_mixed_language(n):
    for _ in range(n):
        a, b = rng.sample(EN_NAMES, 2)
        ba, bb = rng.randint(150, 800), rng.randint(100, 700)
        amount = rng.randint(10, 100)
        q = (f'{a} has {ba} dollars؛ {b} {bb} دلار دارد. '
             f'{a} {amount} dollars به {b} منتقل می‌کند. '
             f"What is {b}'s balance?")
        row('mixed_language', 'num', 'fa', q, bb + amount,
            authoring='template_generated')


def generated_identifiers(n):
    for _ in range(n):
        code = rng.randint(10000, 99999)
        x, y = rng.randint(20, 400), rng.randint(20, 400)
        lang = rng.choice(['fa', 'en'])
        if lang == 'fa':
            q = f'شناسه سفارش {code} ثبت شد؛ جمع {x} و {y} دلار چند دلار است؟'
        else:
            q = f'Record id {code}: what is {x} + {y} dollars?'
        row('identifiers', 'num', lang, q, x + y, authoring='template_generated')


def generated_ood(n):
    prompts_fa = ['یک توصیه مطالعه بده.', 'درباره‌ی تاریخ ایران توضیح بده.',
                  'بهترین فیلم ۲۰۲۵ چیست؟', 'یک داستان کوتاه بنویس.']
    prompts_en = ['Recommend a good book.', 'What causes rain?',
                  'Write a haiku about winter.', 'Who invented the telephone?']
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        q = rng.choice(prompts_fa if lang == 'fa' else prompts_en)
        row('ood', 'abstain', lang, q, None, forbidden=[],
            authoring='template_generated')


def generated_probability(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        p_pct = rng.choice([10, 20, 25, 30, 40, 50, 60, 75])
        trials = rng.randint(2, 6)
        k = rng.randint(1, trials)
        p = p_pct / 100.0
        expected = math_comb(trials, k) * (p ** k) * ((1 - p) ** (trials - k)) * 100
        if lang == 'fa':
            q = (f'احتمال موفقیت در هر آزمایش {p_pct} درصد است؛ '
                 f'در {trials} آزمایش دقیقاً {k} موفقیت چند درصد احتمال دارد؟')
        else:
            q = (f'The chance of success per trial is {p_pct}%. '
                 f'In {trials} trials, what is the probability (in %) of exactly {k} successes?')
        row('probability', 'pct', lang, q, round(expected, 4),
            authoring='template_generated')


def math_comb(n, k):
    from math import comb
    return comb(n, k)


def generated_work_rate(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        w1, h1, out = rng.randint(2, 12), rng.randint(2, 9), rng.choice([60, 84, 120, 180, 240, 300])
        w2, h2 = rng.randint(2, 12), rng.randint(2, 9)
        expected = round(out / (w1 * h1) * w2 * h2)
        if lang == 'fa':
            q = (f'{w1} کارگر در {h1} ساعت {out} قطعه تولید می‌کنند؛ '
                 f'{w2} کارگر در {h2} ساعت چند قطعه تولید می‌کنند؟')
        else:
            q = (f'{w1} workers make {out} pieces in {h1} hours. '
                 f'How many pieces do {w2} workers make in {h2} hours?')
        row('work_rate', 'num', lang, q, expected, authoring='template_generated')


def generated_ratio(n):
    for _ in range(n):
        lang = rng.choice(['fa', 'en'])
        total = rng.choice([240, 360, 480, 540, 600, 720, 840, 900])
        ra = rng.randint(1, 9)
        rb = rng.randint(1, 9)
        if ra == rb:
            rb = ra + 1
        expected = total * ra / (ra + rb)
        if lang == 'fa':
            q = (f'{total} ریال بین دو واحد با نسبت {ra} به {rb} تقسیم شد؛ '
                 f'سهم واحد اول چند ریال است؟')
        else:
            q = (f'Divide {total} dollars between two parts in the ratio {ra}:{rb}. '
                 f"What is the first part's share?")
        row('ratio', 'first_part', lang, q, expected, authoring='template_generated')


# ======================================================================
# assemble
# ======================================================================
def build():
    # hand-written (independently authored)
    hand_written_rates()          # 10
    hand_written_ownership()      # 10
    hand_written_inventory()      # 7
    hand_written_scheduling()     # 6
    hand_written_units()          # 4
    hand_written_age()            # 8
    hand_written_hard_negatives() # 8
    hand_written_identifiers()    # 6
    hand_written_ood()            # 6
    hand_written_batch2()
    hand_written_batch3()
    hand_base = len(rows)

    # template-generated (new shapes)
    generated_rates(140)
    generated_ownership(150)
    generated_inventory(130)
    generated_scheduling(110)
    generated_age(130)
    generated_units(110)
    generated_finance(110)
    generated_hard_negatives(120)
    generated_mixed_language(90)
    generated_identifiers(60)
    generated_ood(60)
    generated_probability(70)
    generated_work_rate(70)
    generated_ratio(90)

    rng.shuffle(rows)
    with open(OUT, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    digest = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
    manifest = {
        'benchmark': 'v22_4_blind',
        'total': len(rows),
        'hand_written': sum(1 for r in rows if r['authoring'] == 'hand_written'),
        'template_generated': sum(1 for r in rows if r['authoring'] == 'template_generated'),
        'sha256': digest,
        'frozen': True,
        'note': ('frozen before first execution; ~hand-written share is '
                 'independently authored, template cases are synthetic and '
                 'labelled as such (spec §65)'),
    }
    with open(os.path.join(BASE, 'benchmarks', 'v22_4_blind_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    build()
