"""Benchmark fastticks vs pandas vs a pure-python loop on one full tick day.
Usage: python bench.py path/to/ticks_YYYY-MM-DD.parquet [token]   (default token 256265 = NIFTY 50 spot)"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))
import fastticks as ft, reference as R
from test_fastticks import load_spot


def tm(f, rep=3):
    best = 1e9
    for _ in range(rep):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best


def main(path, token=256265):
    t_ns, x = load_spot(path, token); ts = t_ns / 1e9
    d = np.abs(np.diff(x, prepend=x[0])); m = np.diff(x, prepend=x[0]) > 0
    gap = ft.ewm_halflife(t_ns, x, 60) - ft.ewm_halflife(t_ns, x, 300)
    print(f'{os.path.basename(path)} token {token}: {len(x):,} ticks\n')
    rows = [('ewm_halflife 60s', lambda: ft.ewm_halflife(ts, x, 60), lambda: R.pd_ewm(t_ns, x, 60), lambda: R.py_ewm(t_ns, x, 60)),
            ('rolling_range 60s', lambda: ft.rolling_range(ts, x, 60), lambda: R.pd_rrange(t_ns, x, 60), lambda: R.py_rrange(t_ns, x, 60)),
            ('rolling_quantile 180s q.95', lambda: ft.rolling_quantile(ts, d, 180, .95, 50), lambda: R.pd_rq(t_ns, d, 180, .95, 50), lambda: R.py_rq(t_ns, d, 180, .95, 50)),
            ('rolling_count 60s', lambda: ft.rolling_count(ts, m, 60), lambda: R.pd_rcount(t_ns, m, 60), lambda: R.py_rcount(t_ns, m, 60)),
            ('wave_events', lambda: ft.wave_events(gap), None, lambda: R.wave_events_ref(gap))]
    print(f"{'function':28s} {'C++ ms':>9s} {'pandas ms':>10s} {'python ms':>10s} {'vs pandas':>10s} {'vs python':>10s}")
    for nm, c, p, y in rows:
        tc = tm(c, 7); tp = tm(p) if p else float('nan'); ty = tm(y, 1)
        print(f'{nm:28s} {tc*1e3:9.2f} {tp*1e3:10.2f} {ty*1e3:10.1f} {tp/tc:9.0f}x {ty/tc:9.0f}x')
    # streaming: python calling update() once per tick (what a live loop would do)
    e = ft.EWM(60); q = ft.RollingQuantile(180, .95, 50); r = ft.RollingMinMax(60)
    tl = t_ns.tolist(); xl = x.tolist(); dl = d.tolist()
    t0 = time.perf_counter()
    for a, b, c in zip(tl, xl, dl): e.update_ns(a, b); r.update_ns(a, b); q.update_ns(a, c)
    el = time.perf_counter() - t0
    print(f'\nstreaming EWM + RollingMinMax + RollingQuantile per tick from python: {el / len(tl) * 1e6:.2f} us/tick ({el*1e3:.0f} ms for the day)')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 256265)
