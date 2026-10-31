"""Verification of fastticks.Window (append path vs pandas, simulate vs brute force).

Run:  python tests/test_window.py [path/to/ticks_YYYY-MM-DD.parquet]     or   python -m pytest tests
"""
import os, sys, math
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import fastticks as ft
from test_fastticks import random_case, load_spot, maxdiff

STATS = ['count', 'sum', 'mean', 'var', 'std', 'min', 'max', 'range', 'median', 'q95', 'first', 'last', 'delta', 'slope', 'ema60', 'ema300']


def stats_of(w):
    return [w.count(), w.sum(), w.mean(), w.var(), w.std(), w.min(), w.max(), w.range(), w.median(), w.quantile(0.95),
            w.first(), w.last(), w.delta(), w.slope(), w.ema('60s'), w.ema('300s')]


def stats_of_sim(s):
    return [s.count(), s.sum(), s.mean(), s.var(), s.std(), s.min(), s.max(), s.range(), s.median(), s.quantile(0.95),
            s.first(), s.last(), s.delta(), s.slope(), s.ema('60s'), s.ema('300s')]


def brute_window(t_ns, x, i, W_ns, maxlen=0):
    """exact window values for row i (valid values with t_i - t_j < W, last maxlen of them)."""
    j0 = np.searchsorted(t_ns, t_ns[i] - W_ns, side='right') if W_ns else 0
    tt, vv = t_ns[j0:i + 1], x[j0:i + 1]; m = ~np.isnan(vv); tt, vv = tt[m], vv[m]
    if maxlen: tt, vv = tt[-maxlen:], vv[-maxlen:]
    return tt, vv


def exact_stats(tt, vv):
    n = len(vv)
    if n == 0: return dict(mean=np.nan, std=np.nan, slope=np.nan, first=np.nan, last=np.nan)
    mean = math.fsum(vv) / n
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in vv) / (n - 1)) if n > 1 else np.nan
    if n > 1:
        ts = (tt - tt[0]) * 1e-9; tm = math.fsum(ts) / n; den = math.fsum((a - tm) ** 2 for a in ts)
        slope = math.fsum((a - tm) * (b - mean) for a, b in zip(ts, vv)) / den if den > 0 else np.nan
    else: slope = np.nan
    return dict(mean=mean, std=std, slope=slope, first=vv[0], last=vv[-1])


def append_vs_pandas(t_ns, x, label, rows, W=60):
    w = ft.Window(maxtime=f'{W}s', ema_halflife=['60s', '300s'])
    got = np.empty((len(x), len(STATS)))
    for i in range(len(x)):
        w.append(int(t_ns[i]), x[i]); got[i] = stats_of(w)
    s = pd.Series(x, index=pd.DatetimeIndex(t_ns.astype('datetime64[ns]'))); r = s.rolling(f'{W}s')
    ref = {'count': r.count(), 'sum': r.sum(), 'mean': r.mean(), 'var': r.var(), 'std': r.std(), 'min': r.min(), 'max': r.max(),
           'range': r.max() - r.min(), 'median': r.median(), 'q95': r.quantile(0.95),
           'ema60': s.ewm(halflife='60s', times=s.index).mean(), 'ema300': s.ewm(halflife='300s', times=s.index).mean()}
    for k, v in ref.items():
        md, ok = maxdiff(got[:, STATS.index(k)], v.values); rows.append((label, f'append {k} vs pandas rolling({W}s)/ewm', md, ok))
    flat = (got[:, STATS.index('range')] == 0) & (got[:, STATS.index('count')] >= 2)
    nf = ~flat
    for k in ('var', 'std'):
        md, ok = maxdiff(got[nf, STATS.index(k)], ref[k].values[nf]); rows.append((label, f'append {k} vs pandas, non-flat windows only', md, ok))
    rows.append((label, f'append std on {int(flat.sum())} flat windows (exact 0)', float(np.max(got[flat, STATS.index('std')], initial=0.0)), True))
    # where pandas and fastticks disagree most, ask an exact math.fsum brute force who is right
    dd = np.nan_to_num(np.abs(got[:, STATS.index('std')] - ref['std'].values)); dd[flat] = 0; worst = np.argsort(dd)[-300:]
    eo, ep = [], []
    for i in worst:
        _, vv = brute_window(t_ns, x, i, W * 10**9); vv = [v for v in vv if v == v]
        if len(vv) < 2: continue
        m = math.fsum(vv) / len(vv); ex = math.sqrt(math.fsum((v - m) ** 2 for v in vv) / (len(vv) - 1))
        eo.append(abs(got[i, STATS.index('std')] - ex)); ep.append(abs(ref['std'].values[i] - ex))
    rows.append((label, 'append std vs exact, 300 worst-vs-pandas rows', max(eo, default=0.0), True))
    rows.append((label, '  (pandas std vs exact on those rows, info)', max(ep, default=0.0), True))
    # first / last / delta / slope / mean / std vs exact (fsum) brute force on 3000 sampled rows -> also the drift check
    idx = np.unique(np.r_[np.linspace(0, len(x) - 1, 3000).astype(int), len(x) - 1]); W_ns = W * 10**9
    ex = {k: [] for k in ('mean', 'std', 'slope', 'first', 'last')}
    for i in idx:
        e = exact_stats(*brute_window(t_ns, x, i, W_ns))
        for k in ex: ex[k].append(e[k])
    for k in ex:
        md, ok = maxdiff(got[idx, STATS.index(k)], ex[k]); rows.append((label, f'append {k} vs exact brute force (3000 rows)', md, ok))
    md, ok = maxdiff(got[idx, STATS.index('delta')], np.array(ex['last']) - np.array(ex['first'])); rows.append((label, 'append delta vs brute force', md, ok))
    return got


def maxlen_checks(t_ns, x, label, rows):
    xv = x[~np.isnan(x)]; tv = t_ns[~np.isnan(x)]
    w = ft.Window(maxlen=50); g = []
    for a, b in zip(tv, xv): w.append(int(a), b); g.append((w.mean(), w.std(), w.median(), w.max()))
    g = np.array(g); r = pd.Series(xv).rolling(50, min_periods=1)
    for j, (k, v) in enumerate((('mean', r.mean()), ('std', r.std()), ('median', r.median()), ('max', r.max()))):
        md, ok = maxdiff(g[:, j], v.values); rows.append((label, f'maxlen=50 {k} vs pandas rolling(50)', md, ok))
    w = ft.Window(maxtime=60, maxlen=40); g = []; e = []
    idx = set(np.linspace(0, len(x) - 1, 1500).astype(int).tolist())
    for i in range(len(x)):
        w.append(int(t_ns[i]), x[i])
        if i in idx:
            tt, vv = brute_window(t_ns, x, i, 60 * 10**9, 40)
            g.append((w.mean(), w.median(), w.count())); e.append((vv.mean() if len(vv) else np.nan, np.median(vv) if len(vv) else np.nan, len(vv)))
    g, e = np.array(g), np.array(e)
    for j, k in enumerate(('mean', 'median', 'count')):
        md, ok = maxdiff(g[:, j], e[:, j]); rows.append((label, f'maxtime=60s+maxlen=40 {k} vs brute force', md, ok))


def simulate_checks(t_ns, x, label, rows, seed=0, maxlen=None):
    rng = np.random.default_rng(seed); n = len(x)
    w = ft.Window(maxtime='60s', maxlen=maxlen, ema_halflife=['60s', '300s'])
    stops = sorted(rng.choice(np.arange(50, n - 1), 12, replace=False)); worst_i = np.zeros(len(STATS)); worst_s = np.zeros(len(STATS)); okall = True
    i = 0
    for stop in stops:
        while i <= stop: w.append(int(t_ns[i]), x[i]); i += 1
        base = np.nanmean(x[max(0, stop - 50):stop + 1])
        vals = (base + rng.normal(0, 5, (4, 25))).round(2); vals[0, 0] = np.nan; vals[1, 3] = vals[1, 4]   # a NaN + ties
        lt = w.last_time_ns
        for tmode in ('default', 'scalar', 'array'):
            if tmode == 'default': T = None; Tb = np.full(vals.shape, lt)
            elif tmode == 'scalar': T = lt + int(rng.uniform(0, 90) * 1e9); Tb = np.full(vals.shape, T)
            else: Tb = lt + (rng.uniform(0, 90, vals.shape) * 1e9).astype(np.int64); T = Tb
            S = stats_of_sim(w.simulate(vals, t=T))
            for a, (v, tb) in enumerate(zip(vals.ravel(), Tb.ravel())):
                c = w.copy(); c.append(int(tb), v); b = stats_of(c)
                for k in range(len(STATS)):
                    got = np.asarray(S[k]).ravel()[a]; d = 0.0 if (np.isnan(got) and np.isnan(b[k])) else abs(got - b[k])
                    if STATS[k] == 'slope' and not np.isnan(d): d /= max(1.0, abs(b[k]))   # relative: 2 ticks ms apart -> slope ~1e4
                    if np.isnan(d): okall = False
                    else: worst_i[k] = max(worst_i[k], d)
        # scalar input -> scalar output
        s1 = w.simulate(24.0); c = w.copy(); c.append(lt, 24.0)
        assert np.ndim(s1.mean()) == 0 and abs(s1.mean() - c.mean()) < 1e-9 and s1.median() == c.median()
        # sequential vs brute force
        seqv = (base + rng.normal(0, 5, 40)).round(2); dt = float(rng.uniform(0.2, 5))
        S = stats_of_sim(w.simulate(seqv, sequential=True, dt=dt, q=[0.95]))
        c = w.copy()
        for a, v in enumerate(seqv):
            c.append(lt + int(round((a + 1) * dt * 1e9)), v); b = stats_of(c)
            for k in range(len(STATS)):
                got = np.asarray(S[k])[a]; d = 0.0 if (np.isnan(got) and np.isnan(b[k])) else abs(got - b[k])
                if np.isnan(d): okall = False
                else: worst_s[k] = max(worst_s[k], d)
    tag = f' maxlen={maxlen}' if maxlen else ''
    for k, nm in enumerate(STATS):
        rows.append((label, f'simulate{tag} {nm} vs copy+append (independent)', worst_i[k], okall))
        rows.append((label, f'simulate{tag} {nm} vs copy+append (sequential)', worst_s[k], okall))
    return rows


TOL = {'var': 1e-6, 'std': 1e-7, 'slope': 1e-8}
# pandas' own rolling var/std is off by up to ~1e-4 on long flat stretches (e.g. after the close, true std 0.0,
# pandas 8.7e-05); fastticks matches the exact (math.fsum) brute force there, so the pandas std rows get a looser bound
# (8.7e-05 on 1 Oct, 3.8e-04 on 11 Sep, all on flat windows); separate rows check non-flat windows and flat == 0.
PANDAS_STD_TOL = 1e-3


def report(rows):
    print(f"{'data':26s} {'check':58s} {'max_abs_diff':>13s} {'NaN same':>8s}")
    bad = []
    for lab, nm, md, ok in rows:
        tol = next((v for k, v in TOL.items() if f' {k} ' in nm + ' '), 1e-8)
        if ('std' in nm or 'var' in nm) and 'vs pandas' in nm: tol = PANDAS_STD_TOL
        if 'info)' in nm: tol = float('inf')
        flag = ok and md <= tol
        if not flag: bad.append((lab, nm, md, ok))
        print(f'{lab:26s} {nm:58s} {md:13.3e} {str(ok):>8s}' + ('' if flag else '   <-- FAIL'))
    print('ALL OK' if not bad else f'{len(bad)} FAILED')
    return not bad


def _rows_random(seed=0, n=8000):
    t, x = random_case(seed, n=n); t_ns = ft.to_ns(t); rows = []
    append_vs_pandas(t_ns, x, f'random seed{seed}', rows); maxlen_checks(t_ns, x, f'random seed{seed}', rows)
    simulate_checks(t_ns, x, f'random seed{seed}', rows, seed); simulate_checks(t_ns, x, f'random seed{seed}', rows, seed, maxlen=30)
    return rows


def ready_checks(t_ns, x, seed=0):
    """Window.ready() / warmup_left() and SimResult.ready() vs the definition and vs copy+append. Returns #checks."""
    rng = np.random.default_rng(seed); n_checks = 0
    configs = [dict(maxtime='60s'), dict(maxlen=50), dict(maxtime='60s', maxlen=40), dict(maxtime='60s', min_count=100),
               dict(maxtime='60s', warmup='5min', ema_halflife='60s'), dict(warmup=30), dict()]
    for cfg in configs:
        w = ft.Window(**cfg)
        req_ns = int(round(ft._window._seconds(cfg['warmup']) * 1e9)) if 'warmup' in cfg else int(round(ft._window._seconds(cfg.get('maxtime')) * 1e9))
        req_n = max(cfg.get('min_count', 0), cfg.get('maxlen', 0))
        assert w.ready() is False and abs(w.warmup_left() - req_ns * 1e-9) < 1e-12
        for rnd in range(2):                                   # second round after reset(): warm-up starts again
            stops = set(rng.choice(np.arange(len(x) // 3), 15, replace=False).tolist()) | {0, 1, 2}
            for i in range(len(x) // 3):
                w.append(int(t_ns[i]), x[i])
                el = int(t_ns[i] - t_ns[0])
                exp = el >= req_ns and w.count() >= req_n
                assert w.ready() == exp, (cfg, i)
                assert abs(w.warmup_left() - max(0, req_ns - el) * 1e-9) < 1e-9
                n_checks += 1
                if i in stops:
                    vals = np.round(x[i] + rng.normal(0, 3, 30), 2); vals[0] = np.nan
                    for T in (None, int(t_ns[i]) + int(rng.uniform(0, 90) * 1e9), int(t_ns[i]) + (rng.uniform(0, 400, 30) * 1e9).astype(np.int64)):
                        r = w.simulate(vals, t=T).ready()
                        Tb = np.broadcast_to(int(t_ns[i]) if T is None else T, vals.shape)
                        for a in range(len(vals)):
                            c = w.copy(); c.append(int(Tb[a]), vals[a]); assert r[a] == c.ready(), (cfg, i, a); n_checks += 1
                    r1 = w.simulate(24.0, t=int(t_ns[i]) + 10**9).ready(); c = w.copy(); c.append(int(t_ns[i]) + 10**9, 24.0)
                    assert isinstance(r1, bool) and r1 == c.ready()
            w.reset(); assert w.ready() is False and w.count() == 0 and abs(w.warmup_left() - req_ns * 1e-9) < 1e-12
    # sequential vs copy+append, and empty-window simulate
    for cfg in configs:
        w = ft.Window(**cfg)
        req_n = max(cfg.get('min_count', 0), cfg.get('maxlen', 0))
        sim0 = w.simulate(np.full(5, 1.0), t=np.arange(5, dtype=np.int64) * 10**9)    # empty window: each tick is the first
        assert not sim0.ready().any() or (req_n <= 1 and 'maxtime' not in cfg and 'warmup' not in cfg)
        for i in range(400): w.append(int(t_ns[i]), x[i])
        for dt in (0.5, 3.0, 30.0):
            vals = np.round(x[400] + rng.normal(0, 3, 60), 2)
            rs = w.simulate(vals, sequential=True, dt=dt).ready(); c = w.copy()
            for a, v in enumerate(vals):
                c.append(w.last_time_ns + int(round((a + 1) * dt * 1e9)), v); assert rs[a] == c.ready(), (cfg, dt, a); n_checks += 1
    return n_checks


def test_ready():
    t_s, x = random_case(3, n=3000)
    assert ready_checks(np.rint(t_s * 1e9).astype(np.int64), x) > 0
    # hand example: maxtime 60s, min_count 3
    w = ft.Window(maxtime='60s', min_count=3)
    w.append(0.0, 1); assert not w.ready() and w.warmup_left() == 60.0
    w.append(59.0, 2); assert not w.ready() and abs(w.warmup_left() - 1.0) < 1e-9
    assert w.simulate([5, 6], t=60.0).ready().tolist() == [False, False]        # elapsed ok, but the 0 s tick expires -> 2 values
    w.append(59.5, 3); assert not w.ready()
    assert w.simulate([5, 6], t=[59.9, 60.0]).ready().tolist() == [False, True]
    w.append(60.0, 4); assert w.ready() and w.warmup_left() == 0.0
    w.reset(); assert not w.ready() and w.warmup_left() == 60.0
    w2 = ft.Window(maxlen=3)
    for i in range(3): assert not w2.ready(); w2.append(float(i), i)
    assert w2.ready() and w2.warmup_left() == 0.0


def test_window_random():
    assert report(_rows_random(7, 4000))


def test_time_inputs():
    import datetime as dt
    w = ft.Window(maxtime=pd.Timedelta('60s'), ema_halflife='60s'); t0 = pd.Timestamp('2026-10-01 09:15:00')
    w.append(t0, 1.0); w.append(t0.to_pydatetime() + dt.timedelta(seconds=30), 2.0); w.append(np.datetime64(t0 + pd.Timedelta('70s')), 3.0)
    w.append(int((t0 + pd.Timedelta('80s')).value), 4.0); w.append((t0 + pd.Timedelta('85s')).value / 1e9, 5.0)
    assert w.count() == 4 and w.first() == 2.0 and w.last() == 5.0, (w.count(), w.first())
    assert ft.Window(maxtime=60).count() == 0 and ft.Window(maxtime=dt.timedelta(minutes=1))._w.maxtime_ns == 60 * 10**9


def test_public_api():
    import datetime as dt, importlib
    w = ft.Window(maxtime='15s'); now = dt.datetime.now()
    assert w.append(now, 1.0) is None and w.extend(np.array([1, 2], dtype='i8') + w.last_time_ns, [2.0, 3.0]) is None and w.reset() is None
    try:
        from fastticks import window  # noqa: F401
        raise AssertionError('fastticks.window must not be importable')
    except ImportError:
        pass
    pub = [n for n in dir(ft) if not n.startswith('_')]
    assert 'window' not in pub and 'np' not in pub, pub
    assert set(ft.__all__) <= set(dir(ft))


if __name__ == '__main__':
    rows = _rows_random(0)
    if len(sys.argv) > 1:
        t_ns, ltp = load_spot(sys.argv[1]); lab = 'real ' + os.path.basename(sys.argv[1])[6:16]
        append_vs_pandas(t_ns, ltp, lab, rows); maxlen_checks(t_ns, ltp, lab, rows)
        simulate_checks(t_ns, ltp, lab, rows, 1); simulate_checks(t_ns, ltp, lab, rows, 2, maxlen=200)
    test_time_inputs(); test_ready(); test_public_api()
    if len(sys.argv) > 1: print(f'ready(): {ready_checks(t_ns, ltp, 1):,} checks on the real day vs definition / copy+append: OK')
    sys.exit(0 if report(rows) else 1)
