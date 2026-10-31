"""Verification of fastticks against pandas / the python reference.

Run:  python tests/test_fastticks.py [path/to/ticks_YYYY-MM-DD.parquet]   (prints max abs diffs)
  or: python -m pytest tests            (random data only)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import fastticks as ft
import reference as R


def maxdiff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    nan_ok = np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a) & ~np.isnan(b)
    return (float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0), nan_ok


def random_case(seed=0, n=20000, nan_frac=0.01):
    rng = np.random.default_rng(seed)
    dt = rng.exponential(0.7, n); dt[rng.random(n) < 0.05] = 0.0            # irregular gaps + duplicate timestamps
    dt[rng.random(n) < 0.002] = 150.0                                         # a few long gaps (> window)
    t_s = 33300.0 + np.cumsum(np.round(dt, 3))                                 # ms resolution, seconds since midnight
    x = 22500 + np.cumsum(rng.normal(0, 1, n)).round(2)
    x[rng.random(n) < nan_frac] = np.nan
    return t_s, x


def check_all(t_in, x, label, rows):
    t_ns = ft.to_ns(t_in)
    xv = np.nan_to_num(x, nan=np.nanmean(x)) if np.isnan(x).any() else x
    cases = []
    for h in (60, 300): cases.append((f'ewm_halflife {h}s', ft.ewm_halflife(t_in, x, h), R.pd_ewm(t_ns, x, h)))
    for w in (60, 180):
        cases += [(f'rolling_max {w}s', ft.rolling_max(t_in, x, w), R.pd_rmax(t_ns, x, w)),
                  (f'rolling_min {w}s', ft.rolling_min(t_in, x, w), R.pd_rmin(t_ns, x, w)),
                  (f'rolling_range {w}s', ft.rolling_range(t_in, x, w), R.pd_rrange(t_ns, x, w))]
    d = np.abs(np.diff(xv, prepend=xv[0]))
    for q, mp in ((0.95, 50), (0.5, 1), (0.05, 10), (0.0, 1), (1.0, 1)):
        cases.append((f'rolling_quantile 180s q={q} mp={mp}', ft.rolling_quantile(t_in, d, 180, q, mp), R.pd_rq(t_ns, d, 180, q, mp)))
    cases.append(('rolling_quantile w/ NaN 180s q=0.95', ft.rolling_quantile(t_in, x, 180, 0.95, 5), R.pd_rq(t_ns, x, 180, 0.95, 5)))
    m = np.diff(xv, prepend=xv[0]) > 0
    for w in (60, 180): cases.append((f'rolling_count {w}s', ft.rolling_count(t_in, m, w), R.pd_rcount(t_ns, m, w)))
    for name, a, b in cases:
        md, nan_ok = maxdiff(a, b); rows.append((label, name, md, nan_ok, len(a)))
    # wave events on the EMA gap (60s - 300s), as in his engine
    gap = R.pd_ewm(t_ns, xv, 60) - R.pd_ewm(t_ns, xv, 300)
    for X, F, Xt in ((2.0, 0.3, 1.0), (2.0, 0.3, 2.0), (1.0, 0.3, 0.5)):
        k, i, l, e = ft.wave_events(gap, X, F, Xt); ref = R.wave_events_ref(gap, X, F, Xt)
        same = len(ref) == len(k) and all((a == b for a, b in zip(ref, zip(k.tolist(), i.tolist(), l.tolist(), e.tolist()))))
        rows.append((label, f'wave_events X={X} F={F} Xt={Xt} ({len(ref)} events)', 0.0 if same else np.inf, same, len(gap)))
    return rows


def check_streaming(t_s, x, rows, label):
    t_ns = ft.to_ns(t_s); xv = np.nan_to_num(x, nan=0.0)
    e = ft.EWM(60); a = np.array([e.update_ns(int(t), v) for t, v in zip(t_ns, x)])
    rows.append((label, 'EWM(60) streaming vs batch', *maxdiff(a, ft.ewm_halflife(t_ns, x, 60)), len(a)))
    m = ft.RollingMinMax(60); a = np.array([m.update_ns(int(t), v)[2] for t, v in zip(t_ns, x)])
    rows.append((label, 'RollingMinMax(60) streaming vs batch', *maxdiff(a, ft.rolling_range(t_ns, x, 60)), len(a)))
    q = ft.RollingQuantile(180, 0.95, 50); a = np.array([q.update_ns(int(t), v) for t, v in zip(t_ns, x)])
    rows.append((label, 'RollingQuantile(180,.95,50) streaming vs batch', *maxdiff(a, ft.rolling_quantile(t_ns, x, 180, 0.95, 50)), len(a)))
    c = ft.RollingCount(60); mk = np.diff(xv, prepend=xv[0]) > 0; a = np.array([c.update_ns(int(t), bool(v)) for t, v in zip(t_ns, mk)])
    rows.append((label, 'RollingCount(60) streaming vs batch', *maxdiff(a, ft.rolling_count(t_ns, mk, 60)), len(a)))
    # float-seconds update() path
    e2 = ft.EWM(60); a2 = np.array([e2.update(float(t) / 1e9 if t > 1e15 else t / 1e9, v) for t, v in zip(t_ns, x)])
    rows.append((label, 'EWM(60).update(seconds) vs batch', *maxdiff(a2, a := ft.ewm_halflife(t_ns, x, 60)), len(a2)))
    gap = np.nan_to_num(ft.ewm_halflife(t_ns, x, 60) - ft.ewm_halflife(t_ns, x, 300))
    w = ft.WaveEvents(2.0, 0.3, 1.0); ev = [r for g in gap if (r := w.update(float(g))) is not None]
    k, i, l, ex = ft.wave_events(gap); same = ev == list(zip(k.tolist(), i.tolist(), l.tolist(), ex.tolist()))
    rows.append((label, f'WaveEvents streaming vs batch ({len(ev)} events)', 0.0 if same else np.inf, same, len(gap)))
    return rows


def report(rows):
    print(f"{'data':28s} {'function':52s} {'max_abs_diff':>13s} {'NaN pos same':>12s} {'n':>8s}")
    for lab, nm, md, ok, n in rows: print(f'{lab:28s} {nm:52s} {md:13.3e} {str(ok):>12s} {n:8d}')
    bad = [r for r in rows if not (r[3] and r[2] <= 1e-9)]
    print('ALL OK' if not bad else f'{len(bad)} FAILED: {bad}')
    return not bad


# ---------------- pytest entry points
def test_random_float_seconds():
    t, x = random_case(1); rows = check_all(t, x, 'random', []); assert all(ok and md <= 1e-9 for _, _, md, ok, _ in rows), rows
def test_random_int_ns():
    t, x = random_case(2); rows = check_all(ft.to_ns(t), x, 'random', []); assert all(ok and md <= 1e-9 for _, _, md, ok, _ in rows), rows
def test_streaming():
    t, x = random_case(3, n=5000); rows = check_streaming(t, x, [], 'random'); assert all(ok and md <= 1e-9 for _, _, md, ok, _ in rows), rows


def load_spot(path, token=256265):
    import pandas as pd
    d = pd.read_parquet(path, columns=['recv_ns', 'token', 'last_price'])
    d = d[d.token == token].sort_values('recv_ns', kind='stable')
    return d.recv_ns.values.view(np.int64), d.last_price.values.astype(float)


if __name__ == '__main__':
    rows = []
    for seed in (0, 1):
        t, x = random_case(seed); check_all(t, x, f'random seed{seed} (float s)', rows)
        check_all(ft.to_ns(t), x, f'random seed{seed} (int ns)', rows)
    t, x = random_case(5, n=5000); check_streaming(t, x, rows, 'random (streaming)')
    if len(sys.argv) > 1:
        t_ns, ltp = load_spot(sys.argv[1]); lab = 'real ' + os.path.basename(sys.argv[1])[6:16]
        check_all(t_ns, ltp, lab + ' (ns)', rows)
        check_all(t_ns / 1e9, ltp, lab + ' (float s)', rows)
        check_streaming(t_ns, ltp, rows, lab + ' (stream)')
    sys.exit(0 if report(rows) else 1)
