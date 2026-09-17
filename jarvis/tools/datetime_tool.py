from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class DateTimeResult:
    time: str
    date: str
    weekday_en: str
    weekday_fa: str


class DateTimeTool:
    WEEKDAYS_FA = (
        "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنج‌شنبه", "جمعه", "شنبه", "یکشنبه"
    )

    @classmethod
    def current(cls) -> DateTimeResult:
        now = datetime.now().astimezone()
        return DateTimeResult(
            now.strftime("%H:%M"),
            now.strftime("%Y-%m-%d"),
            now.strftime("%A"),
            cls.WEEKDAYS_FA[now.weekday()],
        )
