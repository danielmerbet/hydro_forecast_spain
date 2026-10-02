"""Calendar days of a weather-model run (00 or 12 UTC) — shared by scripts 10 and 11.

Every product in this project (ERA5-Land, observations, forecasts) uses UTC
calendar days. A run initialised at hour H covers forecast hours 1..360; the
UTC day d = init date + j is made of the forecast hours (24 j - H, 24 j + 24 - H].
Only complete days inside the 360 hours are kept:
  00 UTC run: days 0..14  -> 15 days starting on the init date
  12 UTC run: days 1..14  -> 14 days starting the day after the init date
(the first 12 h of a 12 UTC run belong to the init date, which is already
covered by the bridge of script 12).
"""
from __future__ import annotations

import datetime as dt

RUN_HOURS = (0, 12)         # runs with a 15-day range in both WeatherNext 3 and AIFS(-ENS)
MAX_HOUR = 360


def parse_init(s: str) -> dt.datetime:
    """'YYYYMMDDHH', 'YYYY-MM-DD' (00 UTC) or 'YYYY-MM-DDTHH[:MM]'."""
    s = s.strip()
    if len(s) == 10 and s.isdigit():
        return dt.datetime.strptime(s, "%Y%m%d%H")
    t = dt.datetime.fromisoformat(s.replace("Z", ""))
    return t.replace(tzinfo=None)


def day_windows(init: dt.datetime, max_days: int = 15) -> list[tuple[dt.date, int, int]]:
    """[(UTC date, first hour exclusive, last hour inclusive)] of the complete days of the run."""
    h0 = init.hour
    if h0 not in RUN_HOURS:
        raise ValueError(f"run hour {h0} not supported (use {RUN_HOURS})")
    out = []
    for j in range(0, 17):
        a, b = 24 * j - h0, 24 * j + 24 - h0
        if a >= 0 and b <= MAX_HOUR:
            out.append((init.date() + dt.timedelta(days=j), a, b))
    return out[:max_days]


def snap_to_run(t: dt.datetime) -> dt.datetime:
    """The latest 00/12 UTC run at or before t."""
    return t.replace(hour=12 if t.hour >= 12 else 0, minute=0, second=0, microsecond=0)
