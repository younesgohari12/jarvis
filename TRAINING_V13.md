# JARVIS v13 — Generalization-100 + Reasoning-100

این مرحله روی release سالم v12 انجام شده و معماری مدل را تغییر نمی‌دهد. هدف، افزایش کیفیت روی دو محور جداست: **Generalization** و **Reasoning**.

## داده‌های جدید

- `datasets/generalization_v13/`: دقیقاً 100 فایل JSONL، مجموع 3,600 نمونهٔ یکتا.
- `datasets/reasoning_v13/`: دقیقاً 100 فایل JSONL، مجموع 3,600 نمونهٔ یکتا.
- `datasets/manifest_v005.json`: دیتاست ترکیبی جدید با 17,529 نمونه.
- concept-groupها به‌صورت اتمی split می‌شوند و leakage بین train/validation/test صفر است.

Generalization شامل تغییر بیان، typo/noise، زبان ترکیبی، constraint-following، distractor robustness، context/reference، negation sensitivity، semantic invariance و composition است.

Reasoning شامل محاسبات چندمرحله‌ای، درصد/نسبت، معادله، دنباله، احتمال، مجموعه، منطق گزاره‌ای، ترتیب قیود، زمان‌بندی، code/list tracing، بودجه و counterfactual است. پاسخ‌های کمی به‌صورت deterministic تولید می‌شوند.

## آموزش

مدل پایه: `models/jarvis_nano_v12.npz`

کاندیدای اصلی با rehearsal داده‌های قبلی آموزش داده شد. برای کاهش drift، release نهایی از interpolation وزن‌های v12 و کاندیدای آموزش‌دیده با ضریب 0.40 استفاده می‌کند.

مدل نهایی: `models/jarvis_nano_v13.npz`

## گیت کیفیت

گیت نهایی در `models/release_gate_v13.json` ثبت شده است. مقایسه با v12 روی holdoutهای مستقل v13 و regression داده‌های v004 انجام می‌شود. release فقط در صورت بهترشدن cross-entropy Generalization و Reasoning و نبود regression معنی‌دار فعال می‌شود.

بازسازی مراحل:

```bash
python training/build_intelligence_200_v13.py
python training/build_dataset_v13.py
python training/tokenize_dataset_v13.py
python training/continue_train_v13_torch.py
python training/blend_v13_candidate.py --alpha 0.4 --output models/jarvis_nano_v13_blend40.npz
python training/evaluate_v13_quality_torch.py --candidate models/jarvis_nano_v13.npz --max-cases 120 --old-cases 120
```
