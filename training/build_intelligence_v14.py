from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SEED = 14092026
VERSION = "dataset_v006"


def h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(blob)
    return h(blob)


def row(kind: str, shard: int, family: int, inp: str, out: str, lang: str, difficulty: str = "medium") -> dict[str, Any]:
    source = f"jarvis-{kind}-v14"
    return {
        "id": f"v14-{kind}-{shard:03d}-{family:02d}",
        "dataset_version": VERSION,
        "origin": source,
        "pretrained_source": None,
        "category": f"v14_{kind}_family_{family:02d}",
        "language": lang,
        "input": inp,
        "output": out,
        "context": [],
        "stage": 11 if kind == "reasoning" else 4,
        "stage_name": "reasoning" if kind == "reasoning" else "instruction_following",
        "metadata": {
            "source": source,
            "quality": "gold-verified",
            "verification": "deterministic_generator_v14",
            "task_type": kind,
            "difficulty": difficulty,
            "risk_level": "L0",
            "requires_tools": False,
            "expected_tool": "",
            "permission_required": False,
            "pretrained_source": None,
            "concept_group": f"v14:{kind}:f{family:02d}:c{shard:03d}",
        },
    }


def fmt(x: float) -> str:
    if math.isclose(x, round(x), abs_tol=1e-10):
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def is_prime(n: int) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    return all(n % d for d in range(3, int(math.isqrt(n)) + 1, 2))


def generalization_examples(i: int) -> list[tuple[str, str, str, str]]:
    # (family_name, input, output, language). Each family varies structure and values.
    a = 17 + i * 3; b = 5 + (i * 7) % 31; c = 2 + (i * 11) % 13
    p = 7 + (i * 13) % 47; n = 120 + i * 9
    x = 11 + i; y = 137 + 2 * i
    d = math.gcd(x, y)
    km = 1.25 + i * 0.17
    cel = -40 + i
    token = f"node-{i:03d}-Az{i%10}"
    nums = [i + 4, i * 2 + 7, i % 17 + 11, i + 29]
    days_fa = ("شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه")
    day_idx = i % 7; off = 2 + i; day_out = days_fa[(day_idx + off) % 7]
    prime_n = 101 + 2 * i
    while prime_n < 400 and prime_n % 2 == 0: prime_n += 1
    aa = 2 + i % 8; xx = 20 + i; bb = 1 + i; cc = aa * xx + bb
    trace_start = 1 + i; trace_steps = 2 + i % 5
    trace = trace_start
    for _ in range(trace_steps): trace = trace * 2 + 1
    qid = 7000 + i
    frac1=(i+2, 2*i+103); frac2=(i+3, 2*i+107)
    left=frac1[0]/frac1[1]; right=frac2[0]/frac2[1]
    cmp = ">" if left > right else ("<" if left < right else "=")
    data = [
        ("distractor_arithmetic", f"علی {a} مهره دارد. رنگ جعبه آبی است و وزنش مهم نیست. {b} مهره می‌گیرد و {c} مهره می‌دهد؛ چند مهره می‌ماند؟", f"{a}+{b}-{c}={a+b-c}. پاسخ: {a+b-c} مهره.", "fa"),
        ("percent_semantics", f"بدون استفاده از وب حساب کن: {p} درصد از {n} چقدر می‌شود؟", f"{p}٪ از {n} = {fmt(p*n/100)}.", "fa"),
        ("ratio_reduction", f"نسبت {x*7} به {y*7} را تا ساده‌ترین حالت کاهش بده.", f"نسبت {x*7}:{y*7} با تقسیم بر {7*d} برابر {x//d}:{y//d} است.", "fa"),
        ("unit_conversion", f"Convert exactly {km:.2f} km to meters; return the number and unit only.", f"{fmt(km*1000)} m", "en"),
        ("temperature", f"دمای {cel} درجهٔ سانتی‌گراد را به فارنهایت تبدیل کن.", f"{cel}°C = {fmt(cel*9/5+32)}°F.", "fa"),
        ("string_length", f"تعداد نویسه‌های رشتهٔ دقیق «{token}» را بدون فاصلهٔ اضافی بگو.", f"رشتهٔ «{token}» دقیقاً {len(token)} نویسه دارد.", "fa"),
        ("string_reverse", f"Reverse this exact token, preserving case: {token}", f"{token[::-1]}", "en"),
        ("list_extrema", f"در فهرست {nums} بزرگ‌ترین و کوچک‌ترین عدد را پیدا کن.", f"کمینه {min(nums)} و بیشینه {max(nums)} است.", "fa"),
        ("sort_unique", f"فهرست [{i%9}, {i%7}, {i%9}, {i%5}, {i%7}, {i+2}] را صعودی و بدون تکرار مرتب کن.", f"{sorted(set([i%9,i%7,i%9,i%5,i%7,i+2]))}", "fa"),
        ("set_intersection", f"A={{ {i}, {i+1}, {i+3}, {i+5} }} و B={{ {i+1}, {i+2}, {i+5}, {i+8} }}. اشتراک را بده.", f"A∩B = {{{i+1}, {i+5}}}.", "fa"),
        ("weekday_offset", f"اگر امروزِ فرضی {days_fa[day_idx]} باشد، {off} روز بعد چه روزی است؟", f"{off} روز بعد از {days_fa[day_idx]}، {day_out} است.", "fa"),
        ("conditional", f"قاعده: اگر n بر 3 بخش‌پذیر بود A، اگر بر 5 بخش‌پذیر بود B و اگر هر دو بود AB. برای n={15*(i+1)} خروجی چیست؟", f"برای n={15*(i+1)} خروجی AB است، چون بر 3 و 5 بخش‌پذیر است.", "fa"),
        ("parity", f"Classify {10_000+i*37} as even or odd and give one-line evidence.", f"{10_000+i*37} is {'even' if (10_000+i*37)%2==0 else 'odd'} because its remainder modulo 2 is {(10_000+i*37)%2}.", "en"),
        ("prime_transfer", f"بررسی کن آیا {prime_n} عدد اول است؛ فقط از مقسوم‌علیه‌ها تا ریشهٔ دوم استفاده کن.", (f"{prime_n} عدد اول است." if is_prime(prime_n) else f"{prime_n} عدد اول نیست؛ یک مقسوم‌علیه آن {next(d for d in range(2,int(math.isqrt(prime_n))+1) if prime_n%d==0)} است."), "fa"),
        ("linear_equation", f"Solve for x: {aa}x + {bb} = {cc}", f"x = {xx}", "en"),
        ("translation_composition", f"به انگلیسی ترجمه کن: سفارش شماره {7000+i} آماده است.", f"Order {7000+i} is ready.", "fa"),
        ("code_trace", f"کد را ذهنی اجرا کن: x={trace_start}; سپس {trace_steps} بار x = 2*x + 1 اجرا می‌شود. x نهایی؟", f"پس از {trace_steps} تکرار از {trace_start}، x={trace} می‌شود.", "fa"),
        ("structured_extract", f'از این رکورد فقط id و score را گزارش کن: {{"id": {qid}, "name": "user-{i}", "score": {80+i%21}, "active": true}}', f"id={qid}, score={80+i%21}", "fa"),
        ("fraction_compare", f"کدام بزرگ‌تر است: {frac1[0]}/{frac1[1]} یا {frac2[0]}/{frac2[1]}؟", f"{frac1[0]}/{frac1[1]} {cmp} {frac2[0]}/{frac2[1]}.", "fa"),
        ("rewrite_constraint", f"این را رسمی و یک‌جمله‌ای کن: سفارشم {qid} فردا میره ارسال", f"سفارش شمارهٔ {qid} فردا ارسال خواهد شد.", "fa"),
    ]
    return data


def reasoning_examples(i: int) -> list[tuple[str, str, str, str]]:
    a=30+i*2; b=4+i%13; c=2+(i*3)%11
    qty=5+i%17; price=12+i*3
    total=qty*price
    dividend=1000+i*17; divisor=7+i%11
    p=10+(i*7)%61; base=200+i*11
    discount=base*p/100
    r1=2+i%7; r2=3+(i*2)%8; units=5+i%9; whole=(r1+r2)*units
    av=[i+3,i+11,i%13+17,i%19+23]
    w1=2+i%4; w2=3+(i*2)%5; v1=50+i; v2=80+i*2
    A=2+i%7; X=4+i%19; B=1+i%9; C=A*X+B
    A2=2+i%5; X2=3+i%13; B2=2+i%7; C2=A2*(X2+B2)
    threshold=20+i; coeff=2+i%5; rhs=coeff*(threshold+1)-1
    g1=24+i*2; g2=36+i*3
    l1=20+i; l2=211+2*i
    pn=97+i*2
    fact=3+i
    comb_n=12+i; comb_k=2+i%4
    perm_n=10+i; perm_k=2+i%3
    toss=5+i; heads=1+(i%4)
    die_target=1+i%5
    seq_start=3+i; seq_diff=2+i%9; seq_n=6+i%5
    geo_start=10+i; geo_ratio=2+i%3; geo_n=5+i%4
    quad_a=1+i; quad_b=3+2*i; # sequence f(k)=a*k^2+b
    age_child=8+i; gap=18+i
    speed=40+i; hours=2+i%5
    rate_a=4+i; rate_b=7+i
    days=("شنبه","یکشنبه","دوشنبه","سه‌شنبه","چهارشنبه","پنجشنبه","جمعه"); di=i%7; off=5+i
    total_minutes=(73*i)%720; hour=total_minutes//60; minute=total_minutes%60
    # clock smaller angle
    ha=30*hour+0.5*minute; ma=6*minute; angle=abs(ha-ma)%360; angle=min(angle,360-angle)
    trace=1+i
    for _ in range(3+i%4): trace=trace*3-1
    prime_flag=is_prime(pn)
    gcdv=math.gcd(g1,g2); lcmv=abs(l1*l2)//math.gcd(l1,l2)
    avg=sum(av)/len(av); weighted=(v1*w1+v2*w2)/(w1+w2)
    work=1/(1/rate_a+1/rate_b)
    arr=[]
    for k in range(1,5): arr.append(quad_a*k*k+quad_b)
    next_quad=quad_a*25+quad_b
    result=[
        ("multi_step", f"سارا {a} امتیاز داشت، {b} گرفت، {c} از دست داد و سپس {b+2} گرفت. امتیاز نهایی؟", f"{a}+{b}-{c}+{b+2}={a+b-c+b+2}. پاسخ {a+b-c+b+2}.", "fa"),
        ("multiplicative_total", f"{qty} بسته داریم و هر بسته {price} قطعه دارد. مجموع قطعات؟", f"{qty}×{price}={total} قطعه.", "fa"),
        ("division_remainder", f"{dividend} را بر {divisor} تقسیم کن و خارج‌قسمت و باقیمانده را بده.", f"{dividend} = {divisor}×{dividend//divisor} + {dividend%divisor}; خارج‌قسمت {dividend//divisor} و باقیمانده {dividend%divisor}.", "fa"),
        ("discount", f"قیمت {base} است و {p}٪ تخفیف می‌خورد. قیمت نهایی؟", f"تخفیف {fmt(discount)} و قیمت نهایی {fmt(base-discount)} است.", "fa"),
        ("increase", f"A value of {base} increases by {p}%. What is the new value?", f"{base} × (1+{p}/100) = {fmt(base*(1+p/100))}.", "en"),
        ("ratio_split", f"عدد {whole} را به نسبت {r1}:{r2} تقسیم کن.", f"مجموع سهم‌ها {r1+r2} است؛ دو بخش برابر {r1*units} و {r2*units} هستند.", "fa"),
        ("average", f"میانگین اعداد {av} را حساب کن.", f"برای اعداد {av}، جمع {sum(av)} است و میانگین {fmt(avg)}.", "fa"),
        ("weighted_average", f"نمره {v1} با وزن {w1} و نمره {v2} با وزن {w2} داریم. میانگین وزنی؟", f"({v1}×{w1}+{v2}×{w2})/({w1}+{w2}) = {fmt(weighted)}.", "fa"),
        ("linear_equation", f"معادله {A}x + {B} = {C} را حل کن.", f"x = ({C}-{B})/{A} = {X}.", "fa"),
        ("two_step_equation", f"Solve {A2}(x + {B2}) = {C2}.", f"From {A2}(x+{B2})={C2}, x+{B2}={C2/A2:g}, so x={X2}.", "en"),
        ("inequality", f"کوچک‌ترین عدد صحیح x که {coeff}x > {rhs} را برقرار کند چیست؟", f"x > {rhs/coeff:g}؛ کوچک‌ترین عدد صحیح {threshold+1} است.", "fa"),
        ("gcd", f"ب.م.م {g1} و {g2} را پیدا کن.", f"gcd({g1},{g2})={gcdv}.", "fa"),
        ("lcm", f"ک.م.م {l1} و {l2} چقدر است؟", f"lcm({l1},{l2})={lcmv}.", "fa"),
        ("prime", f"Is {pn} prime? Verify using trial divisors only up to sqrt(n).", f"{pn} is {'prime' if prime_flag else 'not prime'}" + ("." if prime_flag else f"; divisor {next(d for d in range(2,int(math.isqrt(pn))+1) if pn%d==0)} proves it."), "en"),
        ("factorial", f"{fact}! را دقیق حساب کن.", f"{fact}! = {math.factorial(fact)}.", "fa"),
        ("combinations", f"از {comb_n} عضو، چند زیرمجموعهٔ {comb_k} عضوی می‌توان انتخاب کرد؟", f"C({comb_n},{comb_k})={math.comb(comb_n,comb_k)}.", "fa"),
        ("permutations", f"How many ordered selections of {perm_k} items from {perm_n} distinct items?", f"P({perm_n},{perm_k})={math.perm(perm_n,perm_k)}.", "en"),
        ("coin_exact", f"سکه سالم را {toss} بار می‌اندازیم. احتمال دقیقاً {heads} بار شیر؟", f"C({toss},{heads})/2^{toss} = {math.comb(toss,heads)}/{2**toss} = {fmt(100*math.comb(toss,heads)/(2**toss))}٪.", "fa"),
        ("coin_at_least_one", f"A fair coin is tossed {toss} times. Probability of at least one head?", f"1-(1/2)^{toss} = {fmt(100*(1-1/(2**toss)))}%.", "en"),
        ("die", f"یک تاس سالم را {2+i} بار می‌اندازیم. احتمال اینکه حداقل یک‌بار عددی از 1 تا {die_target} بیاید چقدر است؟", f"در هر پرتاب احتمال موفقیت {die_target}/6 است؛ پس احتمال حداقل یک موفقیت = 1-(1-{die_target}/6)^{2+i} = {fmt(100*(1-(1-die_target/6)**(2+i)))}٪.", "fa"),
        ("arithmetic_sequence", f"دنباله حسابی از {seq_start} با اختلاف {seq_diff} داریم. جملهٔ {seq_n}؟", f"a_{seq_n}={seq_start}+({seq_n}-1)×{seq_diff}={seq_start+(seq_n-1)*seq_diff}.", "fa"),
        ("geometric_sequence", f"Geometric sequence starts at {geo_start} with ratio {geo_ratio}. Find term {geo_n}.", f"a_{geo_n}={geo_start}×{geo_ratio}^{geo_n-1}={geo_start*(geo_ratio**(geo_n-1))}.", "en"),
        ("second_difference", f"دنباله {arr[0]}، {arr[1]}، {arr[2]}، {arr[3]} از قانون درجه‌دو f(k)={quad_a}k²+{quad_b} پیروی می‌کند. جمله بعدی؟", f"f(5)={quad_a}×25+{quad_b}={next_quad}.", "fa"),
        ("age", f"سن فرزند {age_child} سال است و والد {gap} سال بزرگ‌تر است. مجموع سن آن‌ها چند است؟", f"سن والد {age_child+gap} و مجموع سن‌ها {age_child+(age_child+gap)} است.", "fa"),
        ("speed", f"خودرویی با سرعت ثابت {speed} کیلومتر بر ساعت، {hours} ساعت حرکت می‌کند. مسافت؟", f"d=v×t={speed}×{hours}={speed*hours} کیلومتر.", "fa"),
        ("work_rate", f"ماشین A کاری را در {rate_a} ساعت و B در {rate_b} ساعت انجام می‌دهد. باهم چند ساعت؟", f"نرخ مشترک 1/{rate_a}+1/{rate_b} است؛ زمان = {fmt(work)} ساعت.", "fa"),
        ("weekday", f"{off} روز بعد از {days[di]} چه روزی است؟", f"{off} mod 7 = {off%7}؛ پاسخ {days[(di+off)%7]} است.", "fa"),
        ("clock_angle", f"زاویهٔ کوچک‌تر عقربه‌ها در ساعت {hour:02d}:{minute:02d}؟", f"ساعت‌شمار {fmt(ha)}° و دقیقه‌شمار {fmt(ma)}°؛ زاویهٔ کوچک‌تر {fmt(angle)}° است.", "fa"),
        ("syllogism", f"همهٔ A{i}ها B{i} هستند و هیچ B{i}ای C{i} نیست. آیا ممکن است یک A{i}، C{i} باشد؟", f"نه؛ چون A{i}⊆B{i} و B{i}∩C{i}=∅، پس A{i}∩C{i}=∅.", "fa"),
        ("code_trace", f"Trace: x={1+i}; repeat {3+i%4} times: x = 3*x - 1. Final x?", f"Starting at x={1+i} and applying x=3*x-1 for {3+i%4} iterations gives final x={trace}.", "en"),
    ]
    return result


def build_pack(kind: str, count_per_shard: int, generator: Callable[[int], list[tuple[str,str,str,str]]]) -> dict[str, Any]:
    out_dir = ROOT / "datasets" / f"{kind}_v14"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jsonl"): stale.unlink()
    sources=[]; all_rows=[]
    for shard in range(1,101):
        examples=generator(shard)
        if len(examples)!=count_per_shard: raise RuntimeError((kind, shard, len(examples)))
        rows=[]
        for family,(name,inp,out,lang) in enumerate(examples,1):
            rows.append(row(kind, shard, family, inp, out, lang, "hard" if family%5==0 else "medium"))
        file=out_dir/f"{kind[0]}{shard:03d}_mixed.jsonl"
        digest=dump_jsonl(file,rows)
        all_rows.extend(rows)
        sources.append({
            "id": f"{kind[0]}{shard:03d}_mixed", "skill": "mixed_high_diversity",
            "examples": len(rows), "file": file.relative_to(ROOT).as_posix(), "sha256": digest,
            "quality": "gold-verified", "provenance": "project-authored deterministic generator v14",
        })
    inputs=[r['input'].strip().casefold() for r in all_rows]
    outputs=[r['output'].strip().casefold() for r in all_rows]
    pairs=[(a,b) for a,b in zip(inputs,outputs)]
    manifest={
        "format":"jarvis-intelligence-pack-v14", "kind":kind, "version":f"{kind}_100_v14",
        "seed":SEED, "source_dataset_count":100, "examples":len(all_rows),
        "unique_inputs":len(set(inputs)), "unique_outputs":len(set(outputs)), "unique_input_output_pairs":len(set(pairs)),
        "unique_output_ratio":round(len(set(outputs))/len(outputs),6),
        "family_count":count_per_shard,
        "template_index_prefixes":0,
        "quality_gates":{
            "minimum_unique_input_ratio":0.995, "minimum_unique_output_ratio":0.995,
            "deterministic_verification":True, "no_web_sourced_labels":True,
        },
        "sources":sources,
    }
    if len(set(inputs))/len(inputs)<0.995 or len(set(outputs))/len(outputs)<0.995:
        raise RuntimeError(f"{kind} diversity gate failed: {manifest}")
    (ROOT/'datasets'/f'{kind}_100_v14_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


def stable(s:str,n:int=12)->str: return hashlib.sha256(s.encode('utf-8')).hexdigest()[:n]
def split_for_group(group:str)->str:
    b=int(stable(group,8),16)%100
    return 'train' if b<80 else ('validation' if b<90 else 'test')
def read_jsonl(p:Path): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def canonical(r:dict[str,Any])->str:
    return json.dumps({'input':str(r.get('input','')).strip().casefold(),'output':str(r.get('output','')).strip().casefold(),'context':r.get('context',[]) if isinstance(r.get('context',[]),list) else []},ensure_ascii=False,sort_keys=True)


def build_dataset(gm: dict[str,Any], rm: dict[str,Any]) -> dict[str,Any]:
    prior=read_jsonl(ROOT/'datasets'/'cleaned'/'dataset_v004.jsonl')
    for r in prior:
        r['inherited_from']=r.get('inherited_from') or 'dataset_v004'; r['dataset_version']=VERSION
        md=dict(r.get('metadata',{})); concept=str(md.get('concept_group','')).strip() or f'legacy:{stable(canonical(r),16)}'; md['concept_group']=concept; r['metadata']=md
        r['split']=str(r.get('split') or split_for_group(concept))
    added=[]
    for manifest in (gm,rm):
        for src in manifest['sources']:
            path=ROOT/src['file']
            if h(path.read_bytes())!=src['sha256']: raise RuntimeError(f'checksum mismatch: {path}')
            for r in read_jsonl(path):
                concept=str(r['metadata']['concept_group']); r['split']=split_for_group(concept); r['curated_source_id']=src['id']; added.append(r)
    seen=set(); combined=[]; dup=0
    for r in [*prior,*added]:
        fp=h(canonical(r).encode('utf-8'))
        if fp in seen: dup+=1; continue
        seen.add(fp); combined.append(r)
    combined.sort(key=lambda r:(str(r['split']),str(r['id'])))
    splits={s:[r for r in combined if r['split']==s] for s in ('train','validation','test')}
    groups={s:{str(r.get('metadata',{}).get('concept_group','')) for r in vals} for s,vals in splits.items()}
    leakage=sum(len(groups[a]&groups[b]) for a,b in (('train','validation'),('train','test'),('validation','test')))
    if leakage: raise RuntimeError(f'concept leakage {leakage}')
    for sub in ('raw','cleaned','normalized'):
        name='seed_v006.jsonl' if sub=='raw' else 'dataset_v006.jsonl'
        dump_jsonl(ROOT/'datasets'/sub/name,combined)
    for split,vals in splits.items(): dump_jsonl(ROOT/'datasets'/'splits'/f'dataset_v006_{split}.jsonl',vals)
    manifest={
        'format':'jarvis-dataset-manifest-v6','dataset_version':VERSION,'seed':SEED,'total_examples':len(combined),
        'inherited_v004_examples':len(prior),'generalization_examples_added':gm['examples'],'reasoning_examples_added':rm['examples'],
        'generalization_source_dataset_count':gm['source_dataset_count'],'reasoning_source_dataset_count':rm['source_dataset_count'],
        'removed_v13_template_packs':True,'duplicates_removed_during_merge':dup,
        'split_counts':{k:len(v) for k,v in splits.items()},'concept_group_counts':{k:len(v) for k,v in groups.items()},
        'cross_split_concept_leakage':leakage,'languages':dict(Counter(str(r.get('language','unknown')) for r in combined)),
        'origins':dict(Counter(str(r.get('origin','unknown')) for r in combined)),
        'quality_notes':['v004 stable base retained','v13 generalization/reasoning packs are not inherited','v14 packs pass >=99.5% unique-output gate','all v14 labels are deterministic/project-authored','concept groups are atomic across splits'],
    }
    (ROOT/'datasets'/'manifest_v006.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest


def main() -> None:
    gm=build_pack('generalization',20,generalization_examples)
    rm=build_pack('reasoning',30,reasoning_examples)
    dm=build_dataset(gm,rm)
    print(json.dumps({'generalization':{k:gm[k] for k in ('examples','unique_inputs','unique_outputs','unique_output_ratio')},'reasoning':{k:rm[k] for k in ('examples','unique_inputs','unique_outputs','unique_output_ratio')},'dataset':dm},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
