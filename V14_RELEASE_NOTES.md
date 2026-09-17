# JARVIS Intelligence v14 — Release Notes

## هدف
این نسخه به‌جای افزایش نمایشی تعداد دیتاست، گلوگاه Runtime Intelligence را اصلاح می‌کند: Router/Dispatch، Local Reasoning و fallback عصبی قبل از Web.

## نتیجهٔ رفتاری مستقل
- Holdout واقعی و خارج از `dataset_v006`: **30/30 پاس** در v14.
- همان معیار روی v13 دست‌نخورده: **0/30 پاس**.
- تلاش برای Web/Tool در این 30 مسئلهٔ محلی: v13 = **11 مورد**، v14 = **0 مورد**.
- این 30 prompt به‌صورت exact در هیچ split آموزش v006 وجود ندارند.

## Generalization v14
- 100 shard مستقل
- 2,000 مسئله
- 2,000 ورودی یکتا
- 2,000 پاسخ یکتا (**100%**)
- داده‌های Generalization template-heavy نسخه v13 در corpus فعال v006 وارد نشده‌اند.

## Reasoning v14
- 100 shard مستقل
- 3,000 مسئله
- 3,000 ورودی یکتا
- 2,951 پاسخ یکتا (**98.37%**)
- خانواده‌ها شامل arithmetic/multi-step، combinatorics، probability، equations، GCD/LCM، ratios، averages، sequences، time/calendar، code trace و سایر مسائل قابل‌راستی‌آزمایی هستند.
- پک Reasoning تکراری v13 در corpus فعال v006 وارد نشده است.

## Dataset فعال
`dataset_v006`: **15,329 نمونه**؛ مبنای پایدار v004 حفظ شده و 5,000 نمونهٔ v14 جایگزین پک‌های ضعیف v13 شده‌اند. Concept leakage بین train/validation/test = **0**.

## Runtime Intelligence
- `LocalIntelligenceV14` قبل از unknown/web اجرا می‌شود.
- Math / Logic / Probability / Coding / Translation و چندین خانوادهٔ deterministic مستقیماً route می‌شوند.
- `unknown` دیگر به Creative/Story فرستاده نمی‌شود.
- Neural Smart Brain در router uncertainty، قبل از Web فرصت پاسخ دارد و quality gate آن حفظ شده است.
- باگ DialogueSubjectResolver برای ترجمهٔ کامل جمله اصلاح شده است.
- مسیرهای explicit Story / Support / Writing / Rewrite انگلیسی اضافه شده تا regression ایجاد نشود.

## LAB / Training
- دیتاست فعال آموزش: `dataset_v006`.
- LAB پیش از Train، tokenization v14 را اجرا می‌کند.
- دیتاست‌های supervised فعالِ User/Imported به corpus tokenized اضافه می‌شوند.
- Torch trainer دیگر `dataset_v003` را hardcode نمی‌کند.
- Tokenizer: normalized round-trip = **100%**، unknown token rate = **0**.

## Neural model
وزن اصلی همچنان `jarvis_nano_v13.npz` با همان وزن تأییدشده است. عمداً فایل را صرفاً به v14 rename نکرده‌ایم؛ v14 در این release یک ارتقای واقعی Runtime/Reasoning/Training pipeline است، نه یک نام‌گذاری مصنوعی checkpoint.

## Regression
**666/666 تست پاس، 0 خطا.** تست‌ها بعد از آخرین تغییرات به‌صورت ماژول‌به‌ماژول در processهای تازه اجرا شدند تا timeout تجمعی کل suite باعث نتیجهٔ مبهم نشود.
