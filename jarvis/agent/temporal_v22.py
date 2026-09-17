"""JARVIS v22 — Temporal World Model.

Clock time is NOT unrestricted arithmetic: 23:00 + 2h is 01:00 of the next
day (day_offset=+1), never '25'. This model provides typed clock arithmetic
with calendar wrap plus verification support.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Union

UNIT_MINUTES = {'hour': 60, 'hr': 60, 'h': 60, 'minute': 1, 'min': 1,
                'day': 24 * 60, 'روز': 24 * 60, 'ساعت': 60, 'دقیقه': 1}


class TemporalError(ValueError):
    pass


@dataclass
class ClockTime:
    hour: int
    minute: int = 0
    day_offset: int = 0

    def __post_init__(self):
        self.hour = int(self.hour)
        self.minute = int(self.minute)
        self.day_offset = int(self.day_offset)
        if not (0 <= self.hour <= 23 and 0 <= self.minute <= 59):
            raise TemporalError(f'clock_out_of_range: {self.hour:02d}:{self.minute:02d}')

    # ------------------------------------------------------------------
    def absolute_minutes(self) -> int:
        return (self.day_offset * 24 + self.hour) * 60 + self.minute

    def absolute_hours(self) -> float:
        return self.absolute_minutes() / 60.0

    def iso(self) -> str:
        return f'{self.hour:02d}:{self.minute:02d}'

    def to_dict(self) -> dict:
        return {'clock': self.iso(), 'day_offset': self.day_offset,
                'absolute_hours': float(self.absolute_hours())}

    @classmethod
    def from_minutes(cls, total_minutes: int) -> 'ClockTime':
        total_minutes = int(round(total_minutes))
        day_offset, rem = divmod(total_minutes, 24 * 60)
        return cls(rem // 60, rem % 60, day_offset)


@dataclass
class Duration:
    minutes: int

    @classmethod
    def of(cls, value: float, unit: str = 'hour') -> 'Duration':
        factor = UNIT_MINUTES.get(str(unit).strip().lower())
        if factor is None:
            raise TemporalError(f'unknown_duration_unit: {unit}')
        return cls(int(round(float(value) * factor)))

    def to_dict(self) -> dict:
        return {'minutes': self.minutes}

    @property
    def hours(self) -> float:
        return self.minutes / 60.0


def add_duration(clock: ClockTime, *durations: Duration) -> ClockTime:
    """Calendar-aware addition: 23:00 + 2h -> 01:00 (day_offset +1)."""
    total = clock.absolute_minutes()
    for d in durations:
        total += d.minutes
    return ClockTime.from_minutes(total)


def subtract_duration(clock: ClockTime, *durations: Duration) -> ClockTime:
    total = clock.absolute_minutes()
    for d in durations:
        total -= d.minutes
    return ClockTime.from_minutes(total)


def parse_clock_time(value: Union[str, float, int]) -> ClockTime:
    """Parse '23:00' | '23' | 23.5 -> ClockTime."""
    if isinstance(value, str):
        s = value.strip().translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789'))
        if ':' in s:
            h, m = s.split(':', 1)
            return ClockTime(int(h), int(m))
        return ClockTime(int(float(s)), int(round((float(s) % 1) * 60)))
    f = float(value)
    return ClockTime(int(f), int(round((f % 1) * 60)))


def parse_duration(value: float, unit: str = 'hour') -> Duration:
    return Duration.of(value, unit)


def recompute_from_source(start: float, durations: list) -> ClockTime:
    """Independent temporal witness: start + Σ durations, calendar-aware."""
    clock = parse_clock_time(start)
    ds = [d if isinstance(d, Duration) else Duration.of(float(d), 'hour') for d in durations]
    return add_duration(clock, *ds)
