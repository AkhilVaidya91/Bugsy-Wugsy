"""Session windows, same idea as the Mothbox: weekdays x hours x minutes, each lasting runtime_min."""
from __future__ import annotations

from datetime import datetime, timedelta


class Schedule:
    def __init__(self, cfg):
        s = cfg.schedule
        self.enabled = bool(s.enabled)
        self.weekdays = [int(d) for d in s.weekdays]      # 1 = Monday ... 7 = Sunday
        self.hours = [int(h) for h in s.hours]
        self.minutes = [int(m) for m in s.minutes]
        self.runtime = int(s.runtime_min)

    def _starts(self, day):
        wd = day.isoweekday()
        if wd not in self.weekdays:
            return
        for h in self.hours:
            for m in self.minutes:
                yield datetime.combine(day, datetime.min.time()).replace(hour=h, minute=m)

    def status(self, now: datetime | None = None):
        """Returns (active, window_end_or_None)."""
        now = now or datetime.now()
        if not self.enabled:
            return True, None
        for days_back in (0, 1):                 # yesterday too, for sessions that cross midnight
            day = (now - timedelta(days=days_back)).date()
            for start in self._starts(day):
                end = start + timedelta(minutes=self.runtime)
                if start <= now < end:
                    return True, end
        return False, None

    def next_start(self, now: datetime | None = None):
        now = now or datetime.now()
        best = None
        for d in range(0, 9):
            day = (now + timedelta(days=d)).date()
            for start in self._starts(day):
                if start > now and (best is None or start < best):
                    best = start
        return best
