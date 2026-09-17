# نسخهٔ ارتقایافتهٔ JARVIS v20

این پوشه، پروژهٔ کامل ارتقایافته از v19 است. برای اجرا از همان راه‌اندازهای قبلی پروژه استفاده کنید؛ رابط کاربری و مدل اصلی قبلی حفظ شده‌اند. مسیر استدلال اکنون LocalIntelligenceV20 را بارگذاری می‌کند.

وابستگی اجرای مدل‌های جدید فقط NumPy است؛ scipy و scikit-learn فقط برای آموزش لازم‌اند. نصب وابستگی‌های پایه با `python -m pip install -r requirements-runtime.txt` انجام می‌شود. برای بازآموزی، `requirements-training.txt` نیز لازم است.

بازسازی داده و مدل‌ها:

```bash
python training/build_v20_data.py
python training/build_v20_auxiliary.py
python training/train_v20.py
python training/audit_v20_auxiliary.py
```

آزمون:

```bash
python -m pytest -q
```

اجرای همان بنچمارک ثبت‌شده:

```bash
python benchmarks/run_fresh_v20.py . benchmarks/fresh_v20_clean.jsonl reports/v20/reproduced_benchmark.json
```

این دستور همهٔ ابزارهای خارجی و حافظهٔ مکالمه را برای مقایسه خاموش می‌کند. سؤال‌های نیازمند وب فقط از نظر تصمیم تازگی بررسی می‌شوند؛ ادعای تست زندهٔ اینترنت وجود ندارد.

نتیجه‌ها و محدودیت‌ها در `reports/v20/REPORT_FA.md` و وضعیت تک‌تک مراحل در `reports/v20/phase_status.json` آمده‌اند. مدل‌های جدید کوچک و طبقه‌بند هستند؛ مدل زبانی اصلی دوباره آموزش ندیده است. از وجود پنج checkpoint نباید نتیجه گرفت یک مدل زبانی بزرگ یا هوش عمومی جدید ساخته شده است.

داده‌های Holdout را برای آموزش استفاده نکنید. فایل SHA256SUMS.json برای بررسی تمام فایل‌های بسته وجود دارد.
