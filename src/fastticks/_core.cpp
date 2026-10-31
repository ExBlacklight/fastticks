// fastticks: fast, causal, stateful time-series functions for tick data (pybind11).
// Times are int64 nanoseconds internally. The Python wrapper converts float64 seconds,
// datetime64 arrays and pandas DatetimeIndex/Series into int64 ns before calling here.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <string>

namespace py = pybind11;
using i64 = std::int64_t;
static const double NaN = std::numeric_limits<double>::quiet_NaN();

static i64 sec_to_ns(double s) { return (i64)std::llround(s * 1e9); }

// ---------------------------------------------------------------- EWM (pandas ewm(halflife=..., times=...).mean(), adjust=True)
// Replicates pandas' cython ewm loop: com = 1, old_wt *= 0.5 ** (dt_ns / halflife_ns), done per row
// (also across NaN rows, ignore_na=False), with the "weighted != cur" guard.
struct EWM {
    double hl_ns, factor = 0.5; bool adjust = true, ignore_na = false; i64 minp;
    bool started = false; double weighted = NaN, old_wt = 1.0, last_t = 0; i64 nobs = 0;
    EWM(double halflife_s, i64 min_periods = 0, bool adjust_ = true, bool ignore_na_ = false)
        : hl_ns((double)sec_to_ns(halflife_s)), adjust(adjust_), ignore_na(ignore_na_), minp(std::max<i64>(min_periods, 0)) {
        if (!(halflife_s > 0)) throw std::invalid_argument("halflife must be > 0");
        double alpha = 1.0 / (1.0 + 1.0); factor = 1.0 - alpha;
    }
    double update_ns(i64 t, double cur) {
        double tf = (double)t; bool obs = cur == cur;
        if (!started) { started = true; weighted = cur; nobs = obs; old_wt = 1.0; last_t = tf; return nobs >= std::max<i64>(minp, 1) ? weighted : NaN; }
        double delta = (tf - last_t) / hl_ns; last_t = tf; nobs += obs;
        double new_wt = adjust ? 1.0 : (1.0 - factor);
        if (weighted == weighted) {
            if (obs || !ignore_na) {
                old_wt *= std::pow(factor, delta);
                if (obs) {
                    if (weighted != cur) { weighted = old_wt * weighted + new_wt * cur; weighted /= (old_wt + new_wt); }
                    if (adjust) old_wt += new_wt; else old_wt = 1.0;
                }
            }
        } else if (obs) weighted = cur;
        return nobs >= std::max<i64>(minp, 1) ? weighted : NaN;
    }
    double update(double t_s, double x) { return update_ns(sec_to_ns(t_s), x); }
    void reset() { started = false; weighted = NaN; old_wt = 1.0; nobs = 0; }
    double value() const { return nobs >= std::max<i64>(minp, 1) ? weighted : NaN; }
};

// ---------------------------------------------------------------- rolling max / min / range: monotonic deques, window (t - w, t]
struct RollingMinMax {
    i64 w; std::deque<std::pair<i64, double>> qmax, qmin;
    explicit RollingMinMax(double window_s) : w(sec_to_ns(window_s)) { if (w <= 0) throw std::invalid_argument("window must be > 0"); }
    void push(i64 t, double x) {
        while (!qmax.empty() && t - qmax.front().first >= w) qmax.pop_front();
        while (!qmin.empty() && t - qmin.front().first >= w) qmin.pop_front();
        if (x == x) {
            while (!qmax.empty() && qmax.back().second <= x) qmax.pop_back();
            qmax.emplace_back(t, x);
            while (!qmin.empty() && qmin.back().second >= x) qmin.pop_back();
            qmin.emplace_back(t, x);
        }
    }
    double max() const { return qmax.empty() ? NaN : qmax.front().second; }
    double min() const { return qmin.empty() ? NaN : qmin.front().second; }
    double range() const { return qmax.empty() ? NaN : qmax.front().second - qmin.front().second; }
    std::tuple<double, double, double> update_ns(i64 t, double x) { push(t, x); return {max(), min(), range()}; }
    std::tuple<double, double, double> update(double t_s, double x) { return update_ns(sec_to_ns(t_s), x); }
    void reset() { qmax.clear(); qmin.clear(); }
};

// ---------------------------------------------------------------- rolling quantile: sorted vector (binary insert / erase), pandas 'linear'
struct RollingQuantile {
    i64 w; double q; i64 minp; std::deque<std::pair<i64, double>> win; std::vector<double> sorted;
    RollingQuantile(double window_s, double q_, i64 min_periods = 1) : w(sec_to_ns(window_s)), q(q_), minp(min_periods) {
        if (w <= 0) throw std::invalid_argument("window must be > 0");
        if (!(q >= 0 && q <= 1)) throw std::invalid_argument("q must be in [0, 1]");
    }
    double value() const {
        i64 nobs = (i64)sorted.size();
        if (nobs < minp || nobs == 0) return NaN;
        if (nobs == 1) return sorted[0];
        double iwf = q * (double)(nobs - 1); i64 idx = (i64)iwf;
        if ((double)idx == iwf) return sorted[idx];
        double lo = sorted[idx], hi = sorted[idx + 1];
        return lo + (hi - lo) * (iwf - (double)idx);
    }
    double update_ns(i64 t, double x) {
        while (!win.empty() && t - win.front().first >= w) {
            double v = win.front().second; win.pop_front();
            sorted.erase(std::lower_bound(sorted.begin(), sorted.end(), v));
        }
        if (x == x) { win.emplace_back(t, x); sorted.insert(std::upper_bound(sorted.begin(), sorted.end(), x), x); }
        return value();
    }
    double update(double t_s, double x) { return update_ns(sec_to_ns(t_s), x); }
    i64 nobs() const { return (i64)sorted.size(); }
    void reset() { win.clear(); sorted.clear(); }
};

// ---------------------------------------------------------------- rolling count of True in (t - w, t]
struct RollingCount {
    i64 w; std::deque<i64> hits; std::deque<i64> all; i64 cnt = 0;
    explicit RollingCount(double window_s) : w(sec_to_ns(window_s)) { if (w <= 0) throw std::invalid_argument("window must be > 0"); }
    i64 update_ns(i64 t, bool m) {
        while (!hits.empty() && t - hits.front() >= w) hits.pop_front();
        if (m) hits.push_back(t);
        return (i64)hits.size();
    }
    i64 update(double t_s, bool m) { return update_ns(sec_to_ns(t_s), m); }
    void reset() { hits.clear(); }
};

// ---------------------------------------------------------------- wave events: zig-zag state machine on the gap (exact port of the reference)
// kind 1 = peak confirmed (PKC, level = gap at the peak), 2 = trough confirmed (TRC, level = gap at the low).
struct WaveEvents {
    // Only the gap values at the current trough / peak / low are kept (O(1) memory per stream).
    double X, F, Xt; i64 i = -1, tr = 0, pk = -1, lo = 0; double gtr = NaN, gpk = NaN, glo = NaN; bool up = true;
    WaveEvents(double X_peak = 2.0, double F_peak = 0.3, double X_trough = 1.0) : X(X_peak), F(F_peak), Xt(X_trough) {}
    // returns (kind, idx_confirm, level, idx_extreme) or nullopt
    std::optional<std::tuple<int, i64, double, i64>> update(double gi) {
        ++i;
        if (i == 0) { tr = 0; gtr = gi; return std::nullopt; }
        if (up) {
            if (pk < 0 || gi > gpk) { if (gi > gtr) { pk = i; gpk = gi; } }
            if (pk < 0 || gpk - gtr < X) { if (gi < gtr) { tr = i; gtr = gi; pk = -1; } return std::nullopt; }
            if (gi <= gpk - std::max(X, F * gpk)) { up = false; lo = i; glo = gi; return std::make_tuple(1, i, gpk, pk); }
        } else {
            if (gi < glo) { lo = i; glo = gi; }
            if (gi >= glo + Xt) { auto e = std::make_tuple(2, i, glo, lo); tr = lo; gtr = glo; pk = i; gpk = gi; up = true; return e; }
        }
        return std::nullopt;
    }
    std::string state() const { return up ? "up" : "down"; }
    void reset() { i = -1; tr = 0; pk = -1; lo = 0; up = true; gtr = gpk = glo = NaN; }
};

#include "window.hpp"

// ---------------------------------------------------------------- batch (array-in / array-out) versions
using arr_d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arr_i = py::array_t<i64, py::array::c_style | py::array::forcecast>;
static void check_len(const arr_i& t, py::ssize_t n) { if (t.size() != n) throw std::invalid_argument("t and x must have the same length"); }
static void check_mono(const i64* t, py::ssize_t n) { for (py::ssize_t k = 1; k < n; ++k) if (t[k] < t[k - 1]) throw std::invalid_argument("times must be non-decreasing"); }

static arr_d ewm_halflife(arr_i t, arr_d x, double halflife_s, i64 min_periods) {
    auto n = x.size(); check_len(t, n); arr_d out(n);
    const i64* tp = t.data(); const double* xp = x.data(); double* op = out.mutable_data();
    { py::gil_scoped_release r; check_mono(tp, n); EWM e(halflife_s, min_periods); for (py::ssize_t k = 0; k < n; ++k) op[k] = e.update_ns(tp[k], xp[k]); }
    return out;
}
static arr_d rolling_mm(arr_i t, arr_d x, double window_s, int which) {
    auto n = x.size(); check_len(t, n); arr_d out(n);
    const i64* tp = t.data(); const double* xp = x.data(); double* op = out.mutable_data();
    { py::gil_scoped_release r; check_mono(tp, n); RollingMinMax m(window_s);
      for (py::ssize_t k = 0; k < n; ++k) { m.push(tp[k], xp[k]); op[k] = which == 0 ? m.max() : which == 1 ? m.min() : m.range(); } }
    return out;
}
static py::tuple rolling_minmax(arr_i t, arr_d x, double window_s) {
    auto n = x.size(); check_len(t, n); arr_d a(n), b(n);
    const i64* tp = t.data(); const double* xp = x.data(); double* ap = a.mutable_data(); double* bp = b.mutable_data();
    { py::gil_scoped_release r; check_mono(tp, n); RollingMinMax m(window_s); for (py::ssize_t k = 0; k < n; ++k) { m.push(tp[k], xp[k]); ap[k] = m.max(); bp[k] = m.min(); } }
    return py::make_tuple(a, b);
}
static arr_d rolling_quantile(arr_i t, arr_d x, double window_s, double q, i64 min_periods) {
    auto n = x.size(); check_len(t, n); arr_d out(n);
    const i64* tp = t.data(); const double* xp = x.data(); double* op = out.mutable_data();
    { py::gil_scoped_release r; check_mono(tp, n); RollingQuantile rq(window_s, q, min_periods); for (py::ssize_t k = 0; k < n; ++k) op[k] = rq.update_ns(tp[k], xp[k]); }
    return out;
}
static arr_i rolling_count(arr_i t, py::array_t<bool, py::array::c_style | py::array::forcecast> m, double window_s) {
    auto n = m.size(); check_len(t, n); arr_i out(n);
    const i64* tp = t.data(); const bool* mp = m.data(); i64* op = out.mutable_data();
    { py::gil_scoped_release r; check_mono(tp, n); RollingCount c(window_s); for (py::ssize_t k = 0; k < n; ++k) op[k] = c.update_ns(tp[k], mp[k]); }
    return out;
}
static py::tuple wave_events(arr_d gap, double X_peak, double F_peak, double X_trough) {
    auto n = gap.size(); const double* gp = gap.data();
    std::vector<int> kind; std::vector<i64> idx, ext; std::vector<double> lvl;
    { py::gil_scoped_release r; WaveEvents w(X_peak, F_peak, X_trough);
      for (py::ssize_t k = 0; k < n; ++k) if (auto e = w.update(gp[k])) { kind.push_back(std::get<0>(*e)); idx.push_back(std::get<1>(*e)); lvl.push_back(std::get<2>(*e)); ext.push_back(std::get<3>(*e)); } }
    py::array_t<int8_t> K(kind.size()); arr_i I(idx.size()), X(ext.size()); arr_d L(lvl.size());
    for (size_t k = 0; k < kind.size(); ++k) { K.mutable_data()[k] = (int8_t)kind[k]; I.mutable_data()[k] = idx[k]; L.mutable_data()[k] = lvl[k]; X.mutable_data()[k] = ext[k]; }
    return py::make_tuple(K, I, L, X);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "fastticks C++ core (times are int64 ns here; use the fastticks Python wrappers)";
    m.def("ewm_halflife", &ewm_halflife, py::arg("t_ns"), py::arg("x"), py::arg("halflife_s"), py::arg("min_periods") = 0);
    m.def("rolling_max", [](arr_i t, arr_d x, double w) { return rolling_mm(t, x, w, 0); }, py::arg("t_ns"), py::arg("x"), py::arg("window_s"));
    m.def("rolling_min", [](arr_i t, arr_d x, double w) { return rolling_mm(t, x, w, 1); }, py::arg("t_ns"), py::arg("x"), py::arg("window_s"));
    m.def("rolling_range", [](arr_i t, arr_d x, double w) { return rolling_mm(t, x, w, 2); }, py::arg("t_ns"), py::arg("x"), py::arg("window_s"));
    m.def("rolling_minmax", &rolling_minmax, py::arg("t_ns"), py::arg("x"), py::arg("window_s"));
    m.def("rolling_quantile", &rolling_quantile, py::arg("t_ns"), py::arg("x"), py::arg("window_s"), py::arg("q"), py::arg("min_periods") = 1);
    m.def("rolling_count", &rolling_count, py::arg("t_ns"), py::arg("mask"), py::arg("window_s"));
    m.def("wave_events", &wave_events, py::arg("gap"), py::arg("X_peak") = 2.0, py::arg("F_peak") = 0.3, py::arg("X_trough") = 1.0);

    py::class_<EWM>(m, "EWM", "Streaming time-based EWM mean (pandas adjust=True). update(t_seconds, x) -> mean")
        .def(py::init<double, i64>(), py::arg("halflife_s"), py::arg("min_periods") = 0)
        .def("update", &EWM::update, py::arg("t_s"), py::arg("x")).def("update_ns", &EWM::update_ns, py::arg("t_ns"), py::arg("x"))
        .def_property_readonly("value", &EWM::value).def("reset", &EWM::reset);
    py::class_<RollingMinMax>(m, "RollingMinMax", "Streaming time-window max/min/range. update(t_seconds, x) -> (max, min, range)")
        .def(py::init<double>(), py::arg("window_s"))
        .def("update", &RollingMinMax::update, py::arg("t_s"), py::arg("x")).def("update_ns", &RollingMinMax::update_ns, py::arg("t_ns"), py::arg("x"))
        .def_property_readonly("max", &RollingMinMax::max).def_property_readonly("min", &RollingMinMax::min)
        .def_property_readonly("range", &RollingMinMax::range).def("reset", &RollingMinMax::reset);
    py::class_<RollingQuantile>(m, "RollingQuantile", "Streaming time-window quantile (linear). update(t_seconds, x) -> q-value")
        .def(py::init<double, double, i64>(), py::arg("window_s"), py::arg("q"), py::arg("min_periods") = 1)
        .def("update", &RollingQuantile::update, py::arg("t_s"), py::arg("x")).def("update_ns", &RollingQuantile::update_ns, py::arg("t_ns"), py::arg("x"))
        .def_property_readonly("value", &RollingQuantile::value).def_property_readonly("nobs", &RollingQuantile::nobs).def("reset", &RollingQuantile::reset);
    py::class_<RollingCount>(m, "RollingCount", "Streaming count of True in a time window. update(t_seconds, flag) -> count")
        .def(py::init<double>(), py::arg("window_s"))
        .def("update", &RollingCount::update, py::arg("t_s"), py::arg("flag")).def("update_ns", &RollingCount::update_ns, py::arg("t_ns"), py::arg("flag"))
        .def("reset", &RollingCount::reset);
    py::class_<WaveEvents>(m, "WaveEvents", "Streaming zig-zag wave events. update(gap) -> None or (kind, idx_confirm, level, idx_extreme)")
        .def(py::init<double, double, double>(), py::arg("X_peak") = 2.0, py::arg("F_peak") = 0.3, py::arg("X_trough") = 1.0)
        .def("update", &WaveEvents::update, py::arg("gap")).def_property_readonly("state", &WaveEvents::state).def("reset", &WaveEvents::reset);

    using arr_dd = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using arr_ii = py::array_t<i64, py::array::c_style | py::array::forcecast>;
    py::class_<SimResult>(m, "SimResult", "Raw result of Window.simulate (use the fastticks.Window wrapper)")
        .def_property_readonly("n", [](const SimResult& r) { return r.m; })
        .def("get", [](const SimResult& r, const std::string& k) {
            const std::vector<double>* v = k == "count" ? &r.cnt : k == "sum" ? &r.sum : k == "mean" ? &r.mean : k == "m2" ? &r.m2 : k == "max" ? &r.mx :
                k == "min" ? &r.mn : k == "first" ? &r.first : k == "last" ? &r.last : k == "slope" ? &r.slope : nullptr;
            if (!v) throw std::invalid_argument("unknown field " + k);
            return py::array_t<double>(v->size(), v->data()); })
        .def("ema", [](const SimResult& r, size_t i) { if (i >= r.ema.size()) throw std::out_of_range("no EMA at that index"); return py::array_t<double>(r.ema[i].size(), r.ema[i].data()); })
        .def("quantile", [](const SimResult& r, double qq, i64 minp) { auto v = r.quantile(qq, minp); return py::array_t<double>(v.size(), v.data()); }, py::arg("q"), py::arg("min_periods") = 1);
    py::class_<Window>(m, "Window", "Streaming window core (times in int64 ns; use the fastticks.Window wrapper)")
        .def(py::init<double, i64, std::vector<double>>(), py::arg("maxtime_s") = 0.0, py::arg("maxlen") = 0, py::arg("ema_halflives") = std::vector<double>{})
        .def("append_ns", &Window::append_ns, py::arg("t_ns"), py::arg("x"))
        .def("append", &Window::append, py::arg("t_s"), py::arg("x"))
        .def("extend_ns", [](Window& w, arr_ii t, arr_dd x) { if (t.size() != x.size()) throw std::invalid_argument("length mismatch");
            const i64* tp = t.data(); const double* xp = x.data(); auto n = x.size(); py::gil_scoped_release r; w.extend_ns(tp, xp, n); })
        .def("copy", [](const Window& w) { return Window(w); })
        .def("reset", &Window::reset)
        .def("count", &Window::count).def("sum", &Window::sum).def("mean", &Window::mean)
        .def("var", &Window::var, py::arg("ddof") = 1).def("std", &Window::std_, py::arg("ddof") = 1)
        .def("max", &Window::max).def("min", &Window::min).def("range", &Window::range)
        .def("quantile", &Window::quantile, py::arg("q"), py::arg("min_periods") = 1).def("median", &Window::median)
        .def("first", &Window::first).def("last", &Window::last).def("delta", &Window::delta).def("slope", &Window::slope)
        .def("ema", &Window::ema, py::arg("i") = 0)
        .def_property_readonly("halflives", [](const Window& w) { return w.hls; })
        .def_property_readonly("maxtime_ns", [](const Window& w) { return w.maxtime; })
        .def_property_readonly("maxlen", [](const Window& w) { return w.maxlen; })
        .def_property_readonly("last_time_ns", &Window::last_time_ns)
        .def_property_readonly("has_time", [](const Window& w) { return w.has_t; })
        .def_property_readonly("first_time_ns", [](const Window& w) { return w.first_t; })
        .def_property_readonly("n_appends", [](const Window& w) { return w.n_appends; })
        .def("values", [](const Window& w) { py::array_t<double> a(w.q.size()); for (size_t i = 0; i < w.q.size(); ++i) a.mutable_data()[i] = w.q[i].v; return a; })
        .def("times_ns", [](const Window& w) { py::array_t<i64> a(w.q.size()); for (size_t i = 0; i < w.q.size(); ++i) a.mutable_data()[i] = w.q[i].t; return a; })
        .def("simulate", [](const Window& w, arr_dd v, py::object t, bool sequential, std::vector<double> qs) {
            const double* vp = v.data(); py::ssize_t m = v.size(); arr_ii ta; const i64* tp = nullptr; py::ssize_t mt = 0;
            if (!t.is_none()) { ta = arr_ii::ensure(t); if (!ta) throw std::invalid_argument("bad t"); mt = ta.size(); tp = ta.data();
                if (mt != 1 && mt != m) throw std::invalid_argument("t must be a scalar or have the same size as the values"); }
            std::unique_ptr<SimResult> r; { py::gil_scoped_release rel; r = w.simulate(vp, m, tp, mt, sequential, qs); } return r; },
            py::arg("values"), py::arg("t_ns") = py::none(), py::arg("sequential") = false, py::arg("q") = std::vector<double>{});
}
