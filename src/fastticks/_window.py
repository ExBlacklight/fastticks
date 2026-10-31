"""Pythonic wrapper around the C++ streaming Window."""
import datetime as _dt
import numbers
import numpy as np
from . import _core

_UNITS = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0, 'sec': 1.0, 'min': 60.0, 'm': 60.0, 'h': 3600.0}


def _seconds(x):
    """'60s' / '1.5min' / '500ms' / 60 / pd.Timedelta / datetime.timedelta / np.timedelta64 -> float seconds."""
    if x is None:
        return 0.0
    if isinstance(x, numbers.Real):
        return float(x)
    if isinstance(x, np.timedelta64):
        return x / np.timedelta64(1, 'ns') * 1e-9
    if isinstance(x, _dt.timedelta):             # also pd.Timedelta (subclass)
        v = getattr(x, 'value', None)
        return v * 1e-9 if isinstance(v, numbers.Integral) else x.total_seconds()
    if isinstance(x, str):
        s = x.strip().lower()
        for u in sorted(_UNITS, key=len, reverse=True):
            if s.endswith(u) and s[:-len(u)].strip().replace('.', '', 1).isdigit():
                return float(s[:-len(u)]) * _UNITS[u]
        import pandas as pd
        return pd.Timedelta(x).value * 1e-9
    raise TypeError(f'cannot read a duration from {x!r}')


def _ns(t):
    """time -> int ns. float = seconds, int = ns, datetime64 / datetime / pd.Timestamp."""
    if isinstance(t, float):
        return int(round(t * 1e9))
    if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
        return int(t)
    if isinstance(t, np.floating):
        return int(round(float(t) * 1e9))
    if isinstance(t, np.datetime64):
        return int(t.astype('datetime64[ns]').astype(np.int64))
    v = getattr(t, 'value', None)                # pd.Timestamp (tz-aware -> UTC ns)
    if isinstance(v, numbers.Integral) and isinstance(t, _dt.datetime):
        return int(v)
    if isinstance(t, _dt.datetime):
        if t.tzinfo is not None:
            t = t.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return int(np.datetime64(t, 'us').astype(np.int64)) * 1000
    raise TypeError(f'cannot read a time from {t!r}')


def _ns_array(t):
    from . import to_ns
    if np.isscalar(t) or isinstance(t, (_dt.datetime, np.datetime64)):
        return np.array([_ns(t)], dtype=np.int64)
    return to_ns(np.asarray(t).ravel() if not hasattr(t, 'tz') else t)


class Window:
    """Streaming window with incrementally maintained statistics.

    Window(maxtime='60s', maxlen=None, ema=None, min_count=None, warmup=None)
      maxtime : '60s' | seconds | pd.Timedelta | None   -> window (t - maxtime, t], like pandas rolling('60s')
      maxlen  : keep at most N values (both limits apply if both given)
      ema     : '60s' or a list ['60s', '300s'] -> time-based EMAs (the value is the halflife) over ALL appended
                ticks, == pandas ewm(halflife='60s', times=...).mean() (adjust=True; not limited to the window).
                Default: one EMA with halflife = maxtime (none for a maxlen-only window).
                ema() returns the first one listed; ema('300s') picks one.
      min_count : ready() also needs at least this many values in the window
      warmup    : '5min' | seconds -> elapsed time (since the first tick after construction/reset) needed by
                  ready(); replaces the default of maxtime (e.g. 3-5x the ema, since an EMA needs a few halflives to settle)
    ready() is True when all apply: elapsed >= warmup (default maxtime), count >= maxlen (if set), count >= min_count.
    """

    def __init__(self, maxtime=None, maxlen=None, ema=None, min_count=None, warmup=None):
        if ema is None: hl = [maxtime] if maxtime is not None and _seconds(maxtime) > 0 else []
        else: hl = list(ema) if isinstance(ema, (list, tuple)) else [ema]
        self._hl = [_seconds(h) for h in hl]
        self._w = _core.Window(_seconds(maxtime), int(maxlen or 0), self._hl)
        self.maxtime, self.maxlen = maxtime, maxlen
        self.min_count, self.warmup = min_count, warmup
        self._req_ns = int(round(_seconds(warmup) * 1e9)) if warmup is not None else self._w.maxtime_ns
        self._req_n = max(int(min_count or 0), int(maxlen or 0))

    # ---- feeding
    def append(self, time, value):
        """Append one tick. time: float seconds | int ns | datetime | pd.Timestamp | np.datetime64."""
        self._w.append_ns(time if type(time) is int else _ns(time), float(value))

    def extend(self, times, values):
        """Append many ticks at once (arrays)."""
        from . import to_ns
        self._w.extend_ns(to_ns(times), np.ascontiguousarray(values, dtype=np.float64))

    def copy(self):
        w = Window.__new__(Window); w.__dict__.update(self.__dict__); w._hl = list(self._hl); w._w = self._w.copy(); return w

    def reset(self):
        self._w.reset()                              # also restarts the warm-up

    # ---- warm-up
    def ready(self):
        """True once the window is no longer warming up (see class doc); False before the first tick."""
        w = self._w
        return bool(w.has_time and w.last_time_ns - w.first_time_ns >= self._req_ns and w.count() >= self._req_n)

    def warmup_left(self):
        """Seconds of warm-up time still needed (0.0 once the time part is met; ready() also checks the counts)."""
        w = self._w
        if not w.has_time: return self._req_ns / 1e9
        return max(0, self._req_ns - (w.last_time_ns - w.first_time_ns)) / 1e9

    # ---- statistics (scalars)
    def __len__(self): return self._w.count()
    def count(self): return self._w.count()
    def sum(self): return self._w.sum()
    def mean(self): return self._w.mean()
    sma = mean                                   # simple moving average over the window == mean
    def var(self, ddof=1): return self._w.var(ddof)
    def std(self, ddof=1): return self._w.std(ddof)
    def max(self): return self._w.max()
    def min(self): return self._w.min()
    def range(self): return self._w.range()
    def median(self): return self._w.median()
    def quantile(self, q, min_periods=1): return self._w.quantile(q, min_periods)
    def first(self): return self._w.first()
    def last(self): return self._w.last()
    def delta(self): return self._w.delta()
    def slope(self): return self._w.slope()       # OLS slope of value vs time, per second
    def ema(self, halflife=None):
        """EMA value. No argument: the first EMA listed (the maxtime one by default); ema('300s') picks one."""
        return self._w.ema(self._ema_index(halflife))
    def values(self): return self._w.values()
    def times(self): return self._w.times_ns()
    @property
    def last_time_ns(self): return self._w.last_time_ns

    def _ema_index(self, halflife):
        if not self._hl:
            raise ValueError("this Window has no EMA (no maxtime and no ema=): create it with e.g. "
                             "Window(maxlen=..., ema='60s') or ema=['60s', '300s']")
        if halflife is None:
            return 0
        h = _seconds(halflife)
        for i, x in enumerate(self._hl):
            if abs(x - h) < 1e-12:
                return i
        have = [f'{x:g}s' for x in self._hl]
        raise ValueError(f"ema({halflife!r}) is not on this Window (it has ema={have}); add it with Window(..., ema=[...])")

    def stats(self):
        d = dict(count=self.count(), mean=self.mean(), std=self.std(), min=self.min(), max=self.max(), range=self.range(),
                 median=self.median(), first=self.first(), last=self.last(), delta=self.delta(), slope=self.slope())
        for h in self._hl: d[f'ema_{h:g}s'] = self._w.ema(self._hl.index(h))
        return d

    def __repr__(self):
        ema = [f'{h:g}s' for h in self._hl]
        return (f'Window(maxtime={self.maxtime!r}, maxlen={self.maxlen!r}, ema={ema}, n={self.count()}, '
                f'mean={self.mean():.6g}, last={self.last():.6g})')

    # ---- what-if
    def simulate(self, values, t=None, sequential=False, dt=None, q=()):
        """What-if of hypothetical next tick(s); the window itself is NOT changed.

        values : scalar or array of any shape.
        t      : time of the hypothetical tick(s): scalar, or array of the same shape. Default: the last appended
                 time (same timestamp, so nothing extra expires). Must be >= the last appended time.
        sequential=False : every value is an independent "what if THIS is the next tick" (result shape == values shape).
        sequential=True  : values are played as consecutive future ticks (result[i] = state after ticks 0..i);
                 give t (array) or dt (seconds between ticks; first tick at last_time + dt). Default: all at the last time.
                 Only median() and quantiles listed in q=[...] are available in this mode.
        """
        arr = np.asarray(values, dtype=np.float64); shape = arr.shape; flat = np.ascontiguousarray(arr.ravel())
        tn = None
        if t is not None:
            tn = _ns_array(t)
            if tn.size not in (1, flat.size):
                raise ValueError('t must be a scalar or have the same size as values')
        elif dt is not None:
            if not sequential:
                tn = np.array([self._w.last_time_ns + int(round(_seconds(dt) * 1e9))], dtype=np.int64)
            else:
                tn = self._w.last_time_ns + np.rint(np.arange(1, flat.size + 1) * _seconds(dt) * 1e9).astype(np.int64)
        r = self._w.simulate(flat, tn, bool(sequential), [float(x) for x in q])
        res = SimResult(r, shape, self, sequential)
        # warm-up state after the hypothetical tick(s)
        w = self._w; m = flat.size
        te = np.full(m, w.last_time_ns, dtype=np.int64) if tn is None else np.broadcast_to(tn, (m,)).astype(np.int64)
        if w.has_time: first = w.first_time_ns
        else: first = te[0] if (sequential and m) else te
        res._ready = ((te - first) >= self._req_ns) & (np.asarray(r.get('count')) >= self._req_n)
        return res


class SimResult:
    """Result of Window.simulate: same method names as Window; scalars for a scalar input, else arrays of the input shape."""

    def __init__(self, r, shape, win, sequential):
        self._r, self._shape, self._win, self.sequential = r, shape, win, sequential

    def _out(self, a):
        a = np.asarray(a)
        return a.reshape(self._shape)[()] if self._shape == () else a.reshape(self._shape)

    def ready(self):                                   # bool (array): warm-up done after the hypothetical tick(s)
        v = self._out(self._ready); return bool(v) if self._shape == () else v
    def count(self): return self._out(self._r.get('count'))
    def sum(self): return self._out(self._r.get('sum'))
    def mean(self): return self._out(self._r.get('mean'))
    sma = mean
    def var(self, ddof=1):
        n = self._r.get('count'); m2 = self._r.get('m2')
        with np.errstate(invalid='ignore', divide='ignore'):
            return self._out(np.where(n - ddof > 0, m2 / (n - ddof), np.nan))
    def std(self, ddof=1): return np.sqrt(self.var(ddof))
    def max(self): return self._out(self._r.get('max'))
    def min(self): return self._out(self._r.get('min'))
    def range(self): return self._out(self._r.get('max') - self._r.get('min'))
    def median(self): return self._out(self._r.quantile(0.5, 1))
    def quantile(self, q, min_periods=1): return self._out(self._r.quantile(float(q), int(min_periods)))
    def first(self): return self._out(self._r.get('first'))
    def last(self): return self._out(self._r.get('last'))
    def delta(self): return self._out(self._r.get('last') - self._r.get('first'))
    def slope(self): return self._out(self._r.get('slope'))
    def ema(self, halflife=None): return self._out(self._r.ema(self._win._ema_index(halflife)))
    def __len__(self): return int(np.prod(self._shape)) if self._shape else 1
    def __repr__(self): return f'SimResult(shape={self._shape}, sequential={self.sequential})'
