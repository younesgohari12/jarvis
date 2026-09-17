from __future__ import annotations


class FailureRecovery:
    """Converts internal failure codes to short, user-safe responses."""

    @staticmethod
    def message(
        code: str,
        language: str,
        *,
        label: str = "",
        tool: str = "",
        missing: tuple[str, ...] = (),
    ) -> str:
        app = label or ("the application" if language == "en" else "برنامه")
        if code == "not_found":
            return (
                f"I couldn't find {app} on this system."
                if language == "en" else f"{app} رو روی این سیستم پیدا نکردم."
            )
        if code in {"not_running", "already_stopped"}:
            return (
                f"{app} is not currently running."
                if language == "en" else f"{app} الان در حال اجرا نیست."
            )
        if code == "still_running":
            return (
                f"I tried to close {app}, but it is still running."
                if language == "en" else f"تلاش کردم {app} رو ببندم، ولی هنوز در حال اجراست."
            )
        if code == "default_browser_unknown":
            return (
                "Windows did not expose a recognizable default browser. Open it once or name the browser explicitly."
                if language == "en" else
                "ویندوز مرورگر پیش‌فرض قابل‌شناسایی برنگرداند؛ یک‌بار مرورگر را باز کن یا اسمش را مستقیم بگو."
            )
        if code == "missing_argument":
            key = missing[0] if missing else ""
            if key == "app":
                if tool == "close_app":
                    return "Which application should I close?" if language == "en" else "کدوم برنامه رو ببندم؟"
                return "Which application should I open?" if language == "en" else "کدوم برنامه رو باز کنم؟"
            if key in {"path", "folder"}:
                return "Which folder should I use?" if language == "en" else "کدوم فولدر رو باز کنم؟"
            if key == "query":
                return "What should I search for?" if language == "en" else "چی رو سرچ کنم؟"
            if key == "url":
                return "Which address should I open?" if language == "en" else "کدوم آدرس رو باز کنم؟"
        if code in {"open_failed", "close_failed", "operation_failed", "empty_result"}:
            return (
                "The action could not be completed."
                if language == "en" else "نتونستم این کار رو کامل انجام بدم."
            )
        return (
            "The action failed. Enable Debug mode for technical details."
            if language == "en"
            else "اجرای این کار موفق نبود. جزئیات فنی فقط در حالت Debug ثبت شده است."
        )
