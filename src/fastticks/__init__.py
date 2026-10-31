"""fastticks - fast, causal, stateful time-series functions for tick data (C++ via pybind11).

Times ``t`` may be any of:
  * float64 seconds (epoch seconds or seconds-since-midnight; rounded to the nearest ns),
  * int64 nanoseconds,
  * numpy datetime64 (any unit) / pandas DatetimeIndex / datetime Series (converted to ns).
Times must be non-decreasing. Values ``x`` are converted to float64; NaN values are skipped the
same way pandas skips them. Every function is causal: output[i] uses rows 0..i only.
"""
import numpy as _np
from . import _core
from ._core import EWM, RollingMinMax, RollingQuantile, RollingCount, WaveEvents

__all__ = ['ewm_halflife', 'rolling_max', 'rolling_min', 'rolling_range', 'rolling_minmax', 'rolling_quantile',
           'rolling_count', 'wave_events', 'to_ns', 'Window', 'SimResult', 'EWM', 'RollingMinMax', 'RollingQuantile', 'RollingCount', 'WaveEvents']
__version__ = '0.2.0'


def to_ns(t):
    """Convert times (float s / int ns / datetime64 / pandas) to a contiguous int64 ns array."""
    if getattr(getattr(t, 'dtype', None), 'tz', None) is not None:      # tz-aware pandas -> UTC instants
        import pandas as pd
        t = pd.DatetimeIndex(t).tz_convert(None)
    a = _np.asarray(t)
    if a.dtype.kind == 'M':
        return _np.ascontiguousarray(a.astype('datetime64[ns]').view(_np.int64))
    if a.dtype.kind in 'iu':
        return _np.ascontiguousarray(a, dtype=_np.int64)
    if a.dtype.kind == 'f':
        if _np.isnan(a).any():
            raise ValueError('times contain NaN')
        return _np.ascontiguousarray(_np.rint(a * 1e9).astype(_np.int64))
    raise TypeError(f'unsupported time dtype {a.dtype}')


def _x(x):
    return _np.ascontiguousarray(_np.asarray(x, dtype=_np.float64))


def ewm_halflife(t, x, halflife_s, min_periods=0):
    """Time-based EWM mean == pandas ``x.ewm(halflife=f'{halflife_s}s', times=t).mean()`` (adjust=True)."""
    return _core.ewm_halflife(to_ns(t), _x(x), float(halflife_s), int(min_periods))


def rolling_max(t, x, window_s):
    """== pandas ``x.rolling(f'{window_s}s').max()`` (window (t-w, t], includes the current row)."""
    return _core.rolling_max(to_ns(t), _x(x), float(window_s))


def rolling_min(t, x, window_s):
    """== pandas ``x.rolling(f'{window_s}s').min()``."""
    return _core.rolling_min(to_ns(t), _x(x), float(window_s))


def rolling_range(t, x, window_s):
    """== pandas ``r = x.rolling(f'{window_s}s'); r.max() - r.min()``."""
    return _core.rolling_range(to_ns(t), _x(x), float(window_s))


def rolling_minmax(t, x, window_s):
    """(max, min) in one pass."""
    return _core.rolling_minmax(to_ns(t), _x(x), float(window_s))


def rolling_quantile(t, x, window_s, q, min_periods=1):
    """== pandas ``x.rolling(f'{window_s}s', min_periods=min_periods).quantile(q)`` (linear interpolation)."""
    return _core.rolling_quantile(to_ns(t), _x(x), float(window_s), float(q), int(min_periods))


def rolling_count(t, mask, window_s):
    """Count of True in (t-w, t] == pandas ``mask.astype(float).rolling(f'{window_s}s').sum()`` (returned as int64)."""
    m = _np.asarray(mask)
    if m.dtype != _np.bool_:
        m = _np.nan_to_num(m.astype(_np.float64)) != 0
    return _core.rolling_count(to_ns(t), _np.ascontiguousarray(m), float(window_s))


def wave_events(gap, X_peak=2.0, F_peak=0.3, X_trough=1.0):
    """Zig-zag wave state machine on the gap.

    Returns (kind, idx_confirm, level, idx_extreme) numpy arrays, one entry per event:
      kind 1 = peak confirmed (PKC), level = gap at the peak, idx_extreme = index of the peak
      kind 2 = trough confirmed (TRC), level = gap at the low, idx_extreme = index of the low
    The event time / spot are ``t[idx_confirm]`` / ``spot_with[idx_confirm]``.
    """
    return _core.wave_events(_x(gap), float(X_peak), float(F_peak), float(X_trough))


from ._window import Window, SimResult  # noqa: E402
