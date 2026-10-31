"""Pure-pandas and pure-python references used by the tests and the benchmark."""
import bisect
from collections import deque
import numpy as np
import pandas as pd


def ser(t_ns, x):
    return pd.Series(np.asarray(x, dtype=float), index=pd.DatetimeIndex(np.asarray(t_ns, dtype='datetime64[ns]')))


# ---- pandas
def pd_ewm(t_ns, x, h):
    s = ser(t_ns, x); return s.ewm(halflife=f'{h}s', times=s.index).mean().values
def pd_rmax(t_ns, x, w): return ser(t_ns, x).rolling(f'{w}s').max().values
def pd_rmin(t_ns, x, w): return ser(t_ns, x).rolling(f'{w}s').min().values
def pd_rrange(t_ns, x, w):
    r = ser(t_ns, x).rolling(f'{w}s'); return (r.max() - r.min()).values
def pd_rq(t_ns, x, w, q, mp=1): return ser(t_ns, x).rolling(f'{w}s', min_periods=mp).quantile(q).values
def pd_rcount(t_ns, m, w): return ser(t_ns, np.asarray(m, dtype=float)).rolling(f'{w}s').sum().values


# ---- pure python loops (same algorithms, for the speed comparison)
def py_ewm(t_ns, x, h):
    hl = float(round(h * 1e9)); out = np.empty(len(x)); w = x[0]; ow = 1.0; out[0] = w
    for i in range(1, len(x)):
        c = x[i]; ow *= 0.5 ** ((float(t_ns[i]) - float(t_ns[i - 1])) / hl)
        if w != c: w = (ow * w + c) / (ow + 1.0)
        ow += 1.0; out[i] = w
    return out
def py_rrange(t_ns, x, w):
    W = round(w * 1e9); qa, qi = deque(), deque(); out = np.empty(len(x))
    for i in range(len(x)):
        t, v = t_ns[i], x[i]
        while qa and t - qa[0][0] >= W: qa.popleft()
        while qi and t - qi[0][0] >= W: qi.popleft()
        while qa and qa[-1][1] <= v: qa.pop()
        qa.append((t, v))
        while qi and qi[-1][1] >= v: qi.pop()
        qi.append((t, v)); out[i] = qa[0][1] - qi[0][1]
    return out
def py_rq(t_ns, x, w, q, mp=1):
    W = round(w * 1e9); win = deque(); srt = []; out = np.empty(len(x))
    for i in range(len(x)):
        t, v = t_ns[i], x[i]
        while win and t - win[0][0] >= W: srt.pop(bisect.bisect_left(srt, win.popleft()[1]))
        win.append((t, v)); bisect.insort(srt, v); n = len(srt)
        if n < mp: out[i] = np.nan; continue
        f = q * (n - 1); k = int(f); out[i] = srt[k] if k == f else srt[k] + (srt[k + 1] - srt[k]) * (f - k)
    return out
def py_rcount(t_ns, m, w):
    W = round(w * 1e9); h = deque(); out = np.empty(len(m), dtype=np.int64)
    for i in range(len(m)):
        t = t_ns[i]
        while h and t - h[0] >= W: h.popleft()
        if m[i]: h.append(t)
        out[i] = len(h)
    return out


# ---- the user's reference wave state machine (verbatim logic), returning kind/idx/level
def wave_events_ref(gap, X_PEAK=2.0, F_PEAK=0.3, X_TROUGH=1.0):
    ev = []; tr = 0; pk = None; state = 'up'; lo = 0
    for i in range(1, len(gap)):
        if state == 'up':
            if pk is None or gap[i] > gap[pk]:
                pk = i if gap[i] > gap[tr] else pk
            if pk is None or gap[pk] - gap[tr] < X_PEAK:
                if gap[i] < gap[tr]: tr = i; pk = None
                continue
            if gap[i] <= gap[pk] - max(X_PEAK, F_PEAK * gap[pk]):
                ev.append((1, i, gap[pk], pk)); state = 'down'; lo = i
        else:
            if gap[i] < gap[lo]: lo = i
            if gap[i] >= gap[lo] + X_TROUGH:
                ev.append((2, i, gap[lo], lo)); tr = lo; pk = i; state = 'up'
    return ev
