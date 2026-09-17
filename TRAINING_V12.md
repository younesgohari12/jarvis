# JARVIS Curated-100 / dataset_v004

این ارتقا برای بهبود درک سؤال، پاسخ فارسی، دستورپذیری، دانش فنی و استدلال سبک ساخته شده است؛ بدون استفاده از وزن‌های pretrained خارجی.

## چه چیزی اضافه شده است؟

- 100 دیتاست مستقل و project-authored در `datasets/curated_v12/`
- 4,800 نمونه‌ی curated یکتا (48 نمونه در هر دیتاست)
- ادغام با 5,529 نمونه‌ی قبلی و ساخت `dataset_v004` با 10,329 نمونه
- split مبتنی بر concept-group با `cross_split_concept_leakage = 0`
- حفظ tokenizer نسخه‌ی v003 برای سازگاری کامل embeddingهای مدل release
- ادامه‌آموزی محافظه‌کارانه از `jarvis_nano_v08.npz` و انتشار `jarvis_nano_v12.npz`
- دیتاست‌های فعالِ واردشده از LAB در `datasets/imported/*.jsonl` نیز هنگام ساخت v004 به داده‌ی آموزش اضافه می‌شوند.

## تمرکز داده‌ها

مجموعه‌ها شامل فارسی معیار و محاوره، نیم‌فاصله و نگارش، Finglish، رفع ابهام، پیروی از محدودیت‌های دستور، خلاصه‌سازی، بازنویسی، فارسی↔انگلیسی، مفاهیم برنامه‌نویسی/شبکه/وب/امنیت دفاعی/AI، ریاضی، منطق، احتمال، نسبت، درصد، دنباله‌ها و مسئله‌های reasoning هستند.

## بازسازی داده‌ها

```bash
python training/build_curated_100_v12.py
python training/build_dataset_v12.py
python training/tokenize_dataset_v12.py
```

## ادامه‌آموزی کم‌ریسک

نسخه‌ی release با 20 step کم‌نرخ ساخته شد تا catastrophic forgetting رخ ندهد:

```bash
python training/continue_train_v12_torch.py \
  --steps-per-stage 4 \
  --sequence-length 64 \
  --learning-rate 0.00002 \
  --weight-decay 0.001 \
  --curated-ratio 0.58
```

پس از آموزش، `models/release_gate_v12.json` باید پاس شود. کاندیدای 60-step به علت پسرفت روی تست‌های قدیمی رد شد و به عنوان مدل فعال استفاده نمی‌شود.

## نتایج release gate

- holdout curated: token accuracy کلی `0.305893 -> 0.354419`
- holdout فارسی: token accuracy `0.275162 -> 0.320406`، یعنی `+4.524` واحد درصد
- تست قدیمی v003: token accuracy `0.441933 -> 0.453547`
- تست قدیمی v003: cross-entropy `3.882264 -> 3.883659` (تقریباً ثابت)
- planning accuracy: `0.414079 -> 0.426501`
- knowledge accuracy: `0.358447 -> 0.395833`

جزئیات کامل در `models/evaluation_v12_quality_gate.json` و `models/release_gate_v12.json` ذخیره شده است.

## نکته‌ی مهم درباره‌ی «100 دیتاست»

این‌ها 100 بسته‌ی curated مستقل داخل خود پروژه هستند، نه 100 مخزن خارجی دانلودشده. این انتخاب باعث می‌شود provenance، فرمت، deduplication، split و کیفیت داده تحت کنترل پروژه بماند و داده‌ی benchmark/test خارجی ناخواسته وارد train نشود.
