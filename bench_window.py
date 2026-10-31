"""Benchmark fastticks.Window: per-append cost from Python and simulate cost.
Usage: python bench_window.py path/to/ticks_YYYY-MM-DD.parquet"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))
import fastticks as ft
from test_fastticks import load_spot


def per(f, n, rep=3):
    best = 1e9
    for _ in range(rep):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best / n * 1e6


def main(path):
    t_ns, x = load_spot(path); n = len(x); tl = t_ns.tolist(); ts = (t_ns / 1e9).tolist(); xl = x.tolist()
    pts = [pd.Timestamp(v) for v in t_ns[:20000]]
    mk = lambda: ft.Window(maxtime='60s', ema_halflife=['60s', '300s'])
    print(f'{os.path.basename(path)}: {n:,} spot ticks, Window(maxtime=60s, 2 EMAs)\n\nper-append cost from Python (us/tick):')
    def a_ns():
        w = mk()
        for a, b in zip(tl, xl): w.append(a, b)
    def a_s():
        w = mk()
        for a, b in zip(ts, xl): w.append(a, b)
    def a_ts():
        w = mk()
        for a, b in zip(pts, xl): w.append(a, b)
    def a_core():
        w = mk()._w; ap = w.append_ns
        for a, b in zip(tl, xl): ap(a, b)
    def a_stats():
        w = mk()
        for a, b in zip(tl, xl): w.append(a, b); w.mean(); w.median(); w.ema(); w.std(); w.range()
    def a_ext():
        mk().extend(t_ns, x)
    for nm, f, k in (('append(int ns, ltp)', a_ns, n), ('append(float seconds, ltp)', a_s, n), ('append(pd.Timestamp, ltp)', a_ts, len(pts)),
                     ('core append_ns (no wrapper)', a_core, n), ('append + mean/median/ema/std/range', a_stats, n), ('extend(arrays) whole day', a_ext, n)):
        print(f'  {nm:38s} {per(f, k):7.3f}')
    w = mk(); w.extend(t_ns[:50000], x[:50000]); print(f'\nsimulate on a live window of {w.count()} values (us per call, all stats read back):')
    rng = np.random.default_rng(0)
    for m in (1, 100, 10_000):
        v = x[49999] + rng.normal(0, 5, m)
        def s_ind():
            s = w.simulate(v if m > 1 else float(v[0])); s.mean(); s.median(); s.ema(); s.std(); s.max()
        tt = w.last_time_ns + (rng.uniform(0, 30, m) * 1e9).astype(np.int64)
        def s_t():
            s = w.simulate(v, t=tt); s.mean(); s.median(); s.ema(); s.std(); s.max()
        def s_seq():
            s = w.simulate(v, sequential=True, dt=0.5); s.mean(); s.median(); s.ema(); s.std(); s.max()
        def brute():
            for val in v[:min(m, 200)]:
                c = w.copy(); c.append(w.last_time_ns, val); c.mean(); c.median(); c.ema(); c.std(); c.max()
        b = per(brute, 1, 1) * m / min(m, 200)
        print(f'  m={m:<6d} independent {per(s_ind, 1, 5):10.1f} | t array {per(s_t, 1, 5):10.1f} | sequential dt {per(s_seq, 1, 5):10.1f} | python copy+append loop {b:12.1f}')


if __name__ == '__main__':
    main(sys.argv[1])
