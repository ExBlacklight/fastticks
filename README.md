# fastticks

This is a small C++ library (via pybind11) of fast, **causal**, **stateful** time-series functions for tick data. It works the same way numpy does: the loop runs in C++, and Python passes whole arrays in and gets arrays back. Every function has a batch (array) form and a streaming (tick-by-tick) class, and `Window` bundles all the rolling stats for a live loop with `simulate()` what-ifs. Both run the same C++ code, so research results and a live loop agree bit for bit.

## Install (Linux, e.g. pop-os)

```bash
sudo apt install g++ python3-dev           # compiler + Python headers (one time)
tar xzf fastticks.tgz && cd fastticks
python3 -m venv ~/venv-ft && . ~/venv-ft/bin/activate   # or use your existing venv
pip install .                               # fetches pybind11 automatically (build isolation)
python tests/test_fastticks.py              # self-check vs pandas (needs pandas)
```

- **No venv:** system python on recent Ubuntu/pop-os blocks a plain `pip install` (PEP 668). Use `pip install --user --break-system-packages .` instead.
- **Offline:** run `pip install pybind11 setuptools wheel` first, then `pip install --no-build-isolation .`.
- **Developing:** use `pip install -e .`.

## Times

`t` can be any of:
- **float64 seconds:** epoch seconds or seconds since midnight. They are rounded to the nearest ns.
- **int64 nanoseconds:** for example `df.recv_ns.values.view('i8')`.
- **`datetime64` arrays, a pandas `DatetimeIndex`, or a datetime Series:** any unit. Tz-aware values are converted to UTC.

Internally everything is int64 ns, which is also what pandas uses. Times must be non-decreasing; duplicate timestamps are fine. NaN values in `x` are skipped the same way pandas skips them.

Exact-ns tip: if you have ns timestamps, pass them as int64. Epoch float seconds only resolve about 0.24 µs, so a tick that lands within ~100 ns of a window edge could fall on the other side (see Caveats).

## Functions (batch, O(n) except the quantile)

```python
import numpy as np, fastticks
ts  = df.recv_ns.values.view('i8')        # or float seconds, or df.index
ltp = df.last_price.values

ema60  = fastticks.ewm_halflife(ts, ltp, 60)     # == s.ewm(halflife='60s', times=s.index).mean()
ema300 = fastticks.ewm_halflife(ts, ltp, 300)
gap    = ema60 - ema300

rlo    = fastticks.rolling_range(ts, ltp, 60)    # == s.rolling('60s').max() - s.rolling('60s').min()
hi60   = fastticks.rolling_max(ts, ltp, 60)      # == s.rolling('60s').max()
lo60   = fastticks.rolling_min(ts, ltp, 60)
hi, lo = fastticks.rolling_minmax(ts, ltp, 60)   # both in one pass

dabs   = np.abs(np.diff(ltp, prepend=ltp[0]))
low    = fastticks.rolling_quantile(ts, dabs, 180, 0.95, min_periods=50)   # == s.rolling('180s', min_periods=50).quantile(.95)
low    = np.r_[np.nan, low[:-1]]                                           # your .shift(1)

n1 = fastticks.rolling_count(ts, ev == 1, 180)   # count of True in the last 180 s (int64)
n0 = fastticks.rolling_count(ts, ev == 0, 180)

kind, idx, level, idx_ext = fastticks.wave_events(gap, X_peak=2.0, F_peak=0.3, X_trough=1.0)
#  kind 1 = peak confirmed (PKC), 2 = trough confirmed (TRC); idx = row where it confirmed;
#  level = gap at the peak / low; idx_ext = row of that peak / low
events = [('PKC' if k == 1 else 'TRC', ts[i], l, spot_with[i]) for k, i, l in zip(kind, idx, level)]
```

| function | pandas / reference it matches | cost |
|---|---|---|
| `ewm_halflife(t, x, halflife_s, min_periods=0)` | `x.ewm(halflife=f'{h}s', times=t).mean()` (adjust=True, ignore_na=False) | O(n) |
| `rolling_max / rolling_min / rolling_range(t, x, window_s)` | `x.rolling(f'{w}s').max()/.min()`; the window is (t−w, t] and includes the current row | O(n) (monotonic deques) |
| `rolling_quantile(t, x, window_s, q, min_periods=1)` | `x.rolling(f'{w}s', min_periods=mp).quantile(q)` (linear interpolation) | O(n·log w + n·w memmove) (sorted vector) |
| `rolling_count(t, mask, window_s)` | `mask.astype(float).rolling(f'{w}s').sum()` (returned as int64) | O(n) |
| `wave_events(gap, X_peak, F_peak, X_trough)` | your zig-zag reference loop, event for event | O(n) |

## Streaming (live, tick by tick)

```python
e60, e300 = fastticks.EWM(60), fastticks.EWM(300)
rng = fastticks.RollingMinMax(60)
q95 = fastticks.RollingQuantile(180, 0.95, min_periods=50)
c1  = fastticks.RollingCount(180)
wav = fastticks.WaveEvents(2.0, 0.3, 1.0)

def on_tick(t_sec, ltp, dabs, is_up_event):
    g = e60.update(t_sec, ltp) - e300.update(t_sec, ltp)
    hi, lo, rlo = rng.update(t_sec, ltp)
    low_prev = q95.value                 # read BEFORE update == .shift(1)
    q95.update(t_sec, dabs)
    n1 = c1.update(t_sec, is_up_event)
    ev = wav.update(g)                   # None or (kind, idx_confirm, level, idx_extreme)
```

- **Times:** every class has `update(t_seconds, x)` and `update_ns(t_ns, x)`.
- **State:** current values are readable via `.value`, `.max`, `.min`, `.range` and `.nobs`, and `reset()` clears a class.
- **`WaveEvents`:** keeps O(1) state, and its indices count updates since creation or `reset()`.

## Window (live stats + `simulate` what-ifs)

`Window` holds the last `maxtime` of ticks (and/or the last `maxlen` values) and keeps every stat up to date incrementally in C++. `simulate()` asks "what would the stats be if the next tick were X?" for one value or a whole array, without changing the window.

```python
import numpy as np
from fastticks import Window

window = Window(maxtime='60s', ema=['300s'])   # maxtime: '60s' | 60 | pd.Timedelta; maxlen=N optional
# window.ema()       -> always the window's own EMA: halflife = maxtime (60 s here)
# window.ema('300s') -> an extra one from the ema= list

def on_tick(time, ltp):            # time: datetime / pd.Timestamp / np.datetime64 / float seconds / int ns
    window.append(time, ltp)
    m, med, e60, e300 = window.mean(), window.median(), window.ema(), window.ema('300s')
    hi, lo, sd, sl = window.max(), window.min(), window.std(), window.slope()   # slope = points per second (OLS)

one_val  = 24
one_list = np.random.randint(10, 100, size=[100,])

sim1 = window.simulate(one_val)          # sim1.mean(), sim1.median(), sim1.ema() ... -> scalars
sim2 = window.simulate(one_list)         # sim2.mean(), sim2.median(), ... -> np arrays, same shape as one_list
sim2.mean().shape                        # (100,)

sim3 = window.simulate(one_list, dt='5s')                 # as if they arrived 5 s after the last tick (older ticks expire); t can be an array too
seq  = window.simulate([24, 25, 26], sequential=True, dt='1s', q=[0.9])   # 3 ticks in a row, 1 s apart
seq.mean()                               # state after tick 1, after 1-2, after 1-3
seq.quantile(0.9)                        # sequential mode: median() always, other quantiles only if listed in q
```

### Warm-up: `ready()`

```python
window = Window(maxtime='60s', warmup='5min', min_count=50)   # ema() = 60 s EMA; warm up 5x that

def on_tick(time, ltp):
    window.append(time, ltp)
    if not window.ready():               # still warming up -> don't trade on these stats yet
        return
    ...

window.warmup_left()                     # seconds of warm-up time still needed (0.0 once met)
sim = window.simulate(one_list, dt='5s')
sim.ready()                              # bool array, same shape as one_list (scalar bool for a scalar input)
```

`ready()` becomes True once **all** of these hold:
- **Elapsed time:** the time since the **first tick appended after construction or `reset()`** is ≥ `warmup` if you gave it, otherwise ≥ `maxtime`, so the window has seen one full span. Use `warmup` to wait longer, for example 3–5× the EMA halflife, because an EMA needs a few halflives to settle.
- **maxlen:** if `maxlen` is set, count ≥ `maxlen`. With maxlen only and no maxtime, this is the only rule.
- **min_count:** if `min_count` is set, the current count must also be ≥ `min_count`.

Notes:
- Before the first tick `ready()` is False. With no maxtime, maxlen, warmup or min_count, it turns True on the first tick.
- `reset()` restarts the warm-up.
- A NaN tick still counts as the first tick for elapsed time, but it doesn't add to the count.
- `warmup_left()` covers only the time part. `ready()` can still be False after it reaches 0 if a count requirement isn't met yet.
- `sim.ready()` is the warm-up state just after each hypothetical tick, using its time and the count after expiry. With `sequential=True` it is the state after ticks 0..i.

`append`, `extend` and `reset` return `None`, so nothing gets echoed in the REPL. Import with `from fastticks import Window`; the internal modules are private.

Stats (on `Window` and on a `simulate` result, same names): `ready count sum mean sma var std median quantile(q) max min range first last delta slope ema()` / `ema('300s')`; `window.stats()` gives a dict of all of them, and `values()` / `times()` return the window contents.

- **Window rule:** like pandas `rolling('60s')`, the window is (t − maxtime, t], measured from the latest tick's time. With `maxlen`, at most N values are kept; if you give both, both limits apply.
- **`sma` is the same as `mean`.** `var`/`std` use ddof=1 by default, like pandas. `delta` = last − first. `slope` is the least-squares slope of the price against time in seconds (NaN when fewer than 2 distinct times).
- **`ema()` with no argument always returns the window's own EMA, sized like the window.** It never returns an entry from `ema=`.
  - **maxtime set:** a time EMA with halflife = maxtime, equal to pandas `ewm(halflife='60s', times=...).mean()` (adjust=True).
  - **maxlen only:** a tick EMA with halflife = maxlen ticks, equal to pandas `ewm(halflife=maxlen).mean()` on the appended prices. NaN ticks count as rows, like pandas.
  - **Both set:** maxtime is used.
  - **Neither set:** `ema()` raises an error.

  ```python
  Window(maxtime='60s').ema()             # 60 s EMA
  Window(maxlen=50).ema()                 # 50-tick EMA
  Window(maxtime='15s', ema=['60s', '300s']).ema()   # still the 15 s EMA
  ```
- **`ema('300s')`** reads an extra EMA from the `ema=` list. `ema=` takes `'300s'` or a list; each entry is a time halflife, and numbers mean seconds.
  - `ema('60s')` on a `maxtime='60s'` window also works without listing it, because it is the window's own EMA.
  - Asking for anything else that isn't listed raises a `ValueError` that names the available ones.
- **Scope:** all EMAs run over **every** appended tick, not just the window. `simulate()` results follow the same rules: `sim.ema()` and `sim.ema('300s')`.
- **`stats()` keys:** `'ema'` for the window's own EMA, plus `'ema_300s'` and so on for listed ones.
- **NaN prices** are skipped like pandas does: the time still advances (old ticks expire) but nothing is added, and `maxlen` counts only real values.
- **`simulate` timing:** the default `t` is the last appended time, so nothing extra expires. `t` must not be earlier than the last appended time.
- **Independent mode (default):** each value is its own "what if this is the next tick". It is vectorized in C++, with no copy of the window.
- **`sequential=True`:** the values are played as consecutive ticks; result[i] is the state after ticks 0..i. Give `dt` (spacing) or a `t` array. If you give neither, all ticks land at the last time.
- **Speed (1 Oct 2026, window of ~240 ticks):** `append` takes ~0.27 µs per tick from Python with int ns (~0.37 µs with float seconds, ~0.85 µs with pd.Timestamp). `simulate(100 values)` takes ~15–25 µs with all stats read back, and 10,000 values take ~0.7–1 ms. A Python copy+append loop is about 15–35× slower. Run `python bench_window.py ticks_day.parquet` for your own numbers.
- **Precision:** sums use compensated (Neumaier) running sums and are recomputed from scratch every ~2×window appends, so they do not drift over a day. On 11 Sep and 1 Oct 2026, std/mean/slope over the full day match an exact `math.fsum` brute force to ≤4e-12. A window where every price is the same gives std exactly 0. pandas' own rolling std is off there by up to 3.8e-04, and by up to 5.4e-06 on near-flat windows; that is pandas' error (checked against the exact brute force). The slope switches to an exact two-pass formula when the window's times are tightly clustered, e.g. 2 ticks 0.2 s apart.

## Verification and speed

- **Verification:** `python tests/test_fastticks.py [ticks_day.parquet]` compares every function with pandas and the Python reference, on random irregular data (with duplicates, NaNs and long gaps) and on a real day.
- **Window:** `python tests/test_window.py [ticks_day.parquet]` checks every `Window` stat against pandas rolling/ewm and an exact brute force, and checks `ready()` / `warmup_left()` / `sim.ready()` against the definition and against copy+append (maxtime, maxlen only, both, min_count, warmup, reset). It also checks `simulate` (scalar, array, `t` array, sequential, with and without maxlen) against copy+append.
- **Benchmark:** `python bench.py ticks_day.parquet [token]` times C++ against pandas and a pure-Python loop. `python bench_window.py ticks_day.parquet` times `Window` and `simulate`.

## Caveats

- Results are bit-identical to pandas 3.x when both get the same int64 ns times. The EWM reproduces pandas' own float rounding: pandas turns times into float64 ns before taking differences, and fastticks does the same. Window edges are compared in integer ns, like pandas.
- If you pass float epoch seconds and build the pandas index from the original ns, a tick within ~0.1 µs of a window boundary could be counted differently. On 1 Oct 2026 this happened 0 times. Pass int64 ns to rule it out.
- `rolling_*` use the pandas default `min_periods=1` for time windows. Only `rolling_quantile` and `ewm_halflife` take `min_periods`.
- Only the pandas `'linear'` interpolation is implemented for the quantile.
- `wave_events` compares raw floats exactly like the Python reference, including NaN behaviour: comparisons with NaN are false. Pass a NaN-free gap.
- `Window` quantiles cost O(log n + n memmove) per append because the window is kept in a sorted vector. That is fine for windows up to tens of thousands of ticks.
- In `simulate(..., sequential=True)` only the median and the quantiles listed in `q=[...]` are computed. Any other `quantile()` call raises an error that tells you to pass it in `q`.
