// Window: a streaming time/count window with incrementally maintained statistics, plus
// vectorised "what-if" simulation of hypothetical next ticks. Included from _core.cpp
// (needs i64, NaN and EWM from there).
#pragma once
#include <memory>
#include <map>

struct KSum {   // Neumaier-compensated running sum (supports removal by adding -x)
    double s = 0, c = 0;
    void add(double x) { double t = s + x; if (std::fabs(s) >= std::fabs(x)) c += (s - t) + x; else c += (x - t) + s; s = t; }
    double v() const { return s + c; }
    void clear() { s = c = 0; }
};

struct WElem { i64 t; double v; i64 seq; };

template <class Get>
static double quantile_of(Get get, i64 n, double q, i64 minp) {   // pandas 'linear' rule, same float ops
    if (n <= 0 || n < minp) return NaN;
    if (n == 1) return get(0);
    double iwf = q * (double)(n - 1); i64 idx = (i64)iwf;
    if ((double)idx == iwf) return get(idx);
    double lo = get(idx), hi = get(idx + 1);
    return lo + (hi - lo) * (iwf - (double)idx);
}

struct SimResult;

struct Window {
    i64 maxtime = 0, maxlen = 0;                 // 0 = unlimited
    std::deque<WElem> q, qmax, qmin;             // window (valid values only) + monotonic deques
    std::vector<double> sorted;                  // order statistics of the window
    KSum s1, s2, st, stt, stv;                   // sums of dv, dv^2, dt, dt^2, dt*dv (dv = v - vref, dt = (t - tref) s)
    double vref = 0; i64 tref = 0; i64 seq = 0, ops = 0;
    std::vector<EWM> emas; std::vector<double> hls;
    i64 last_t = 0, first_t = 0; bool has_t = false; i64 n_appends = 0;   // first_t: first append since construction/reset (warm-up)
    double run_v = NaN; i64 run = 0;           // trailing run of equal values -> exact 0 variance (as pandas)

    Window(double maxtime_s, i64 maxlen_, std::vector<double> ema_halflives) : maxlen(maxlen_) {
        if (maxtime_s > 0) maxtime = sec_to_ns(maxtime_s);
        if (maxtime_s < 0 || maxlen_ < 0) throw std::invalid_argument("maxtime / maxlen must be >= 0");
        for (double h : ema_halflives) { emas.emplace_back(h, 0); hls.push_back(h); }
    }

    // ---------------- maintenance
    void add_sums(const WElem& e, double sign) {
        double dv = e.v - vref, dt = (double)(e.t - tref) * 1e-9;
        s1.add(sign * dv); s2.add(sign * dv * dv); st.add(sign * dt); stt.add(sign * dt * dt); stv.add(sign * dt * dv);
    }
    void recompute() {
        s1.clear(); s2.clear(); st.clear(); stt.clear(); stv.clear(); ops = 0;
        if (q.empty()) return;
        KSum m; for (auto& e : q) m.add(e.v);
        vref = m.v() / (double)q.size(); tref = q.front().t;    // re-anchor at the window mean
        for (auto& e : q) add_sums(e, 1.0);
    }
    void pop_front() {
        WElem e = q.front(); q.pop_front();
        sorted.erase(std::lower_bound(sorted.begin(), sorted.end(), e.v));
        if (!qmax.empty() && qmax.front().seq == e.seq) qmax.pop_front();
        if (!qmin.empty() && qmin.front().seq == e.seq) qmin.pop_front();
        if (q.empty()) { s1.clear(); s2.clear(); st.clear(); stt.clear(); stv.clear(); ops = 0; }
        else { add_sums(e, -1.0); ++ops; }
    }
    void expire_time(i64 t) { if (maxtime > 0) while (!q.empty() && t - q.front().t >= maxtime) pop_front(); }
    void append_ns(i64 t, double v) {
        if (has_t && t < last_t) throw std::invalid_argument("Window.append: time went backwards");
        if (!has_t) first_t = t;
        has_t = true; last_t = t; ++n_appends;
        for (auto& e : emas) e.update_ns(t, v);
        expire_time(t);
        if (!(v == v)) return;                                   // NaN: only time/EMA advance
        if (maxlen > 0) while ((i64)q.size() >= maxlen) pop_front();
        if (q.empty()) { vref = v; tref = t; }
        WElem e{t, v, seq++};
        if (v == run_v) ++run; else { run_v = v; run = 1; }
        q.push_back(e); add_sums(e, 1.0); ++ops;
        sorted.insert(std::upper_bound(sorted.begin(), sorted.end(), v), v);
        while (!qmax.empty() && qmax.back().v <= v) qmax.pop_back(); qmax.push_back(e);
        while (!qmin.empty() && qmin.back().v >= v) qmin.pop_back(); qmin.push_back(e);
        if (ops > 2 * (i64)q.size() + 1024) recompute();       // bounded drift, amortised O(1)
    }
    void append(double t_s, double v) { append_ns(sec_to_ns(t_s), v); }
    void extend_ns(const i64* t, const double* v, py::ssize_t n) { for (py::ssize_t k = 0; k < n; ++k) append_ns(t[k], v[k]); }
    void reset() { q.clear(); qmax.clear(); qmin.clear(); sorted.clear(); recompute(); run = 0; run_v = NaN; for (auto& e : emas) e.reset(); has_t = false; n_appends = 0; first_t = 0; }

    // ---------------- statistics of the current window
    i64 count() const { return (i64)q.size(); }
    double sum() const { return q.empty() ? NaN : vref * (double)q.size() + s1.v(); }
    double mean() const { return q.empty() ? NaN : vref + s1.v() / (double)q.size(); }
    double m2() const { i64 n = q.size(); if (n == 0) return NaN; if (run >= n) return 0.0; double a = s1.v(); double r = s2.v() - a * a / (double)n; return r < 0 ? 0.0 : r; }
    double var(i64 ddof) const { i64 n = q.size(); return n - ddof <= 0 ? NaN : m2() / (double)(n - ddof); }
    double std_(i64 ddof) const { return std::sqrt(var(ddof)); }
    double max() const { return qmax.empty() ? NaN : qmax.front().v; }
    double min() const { return qmin.empty() ? NaN : qmin.front().v; }
    double range() const { return qmax.empty() ? NaN : qmax.front().v - qmin.front().v; }
    double quantile(double qq, i64 minp) const { return quantile_of([&](i64 i) { return sorted[i]; }, (i64)sorted.size(), qq, minp); }
    double median() const { return quantile(0.5, 1); }
    double first() const { return q.empty() ? NaN : q.front().v; }
    double last() const { return q.empty() ? NaN : q.back().v; }
    double delta() const { return q.empty() ? NaN : q.back().v - q.front().v; }
    // exact two-pass OLS slope over q[k..end) plus an optional extra point; used when the running sums are ill-conditioned
    double slope_direct(i64 k, bool extra, i64 tx, double vx) const {
        i64 n = (i64)q.size() - k + (extra ? 1 : 0); if (n < 2) return NaN;
        i64 t0 = (k < (i64)q.size()) ? q[k].t : tx;
        KSum st_, sv_;
        for (i64 i = k; i < (i64)q.size(); ++i) { st_.add((double)(q[i].t - t0) * 1e-9); sv_.add(q[i].v); }
        if (extra) { st_.add((double)(tx - t0) * 1e-9); sv_.add(vx); }
        double tm = st_.v() / (double)n, vm = sv_.v() / (double)n; KSum num, den;
        auto acc = [&](i64 t, double v) { double a = (double)(t - t0) * 1e-9 - tm; num.add(a * (v - vm)); den.add(a * a); };
        for (i64 i = k; i < (i64)q.size(); ++i) acc(q[i].t, q[i].v);
        if (extra) acc(tx, vx);
        return den.v() > 0 ? num.v() / den.v() : NaN;
    }
    // slope from running sums; falls back to the exact two-pass form when the times are clustered relative to tref
    static bool slope_ill(double n, double c, double d) { double den = n * d - c * c; return !(den > 1e-4 * n * d); }
    double slope() const {    // OLS slope of value vs time, per second
        double n = (double)q.size(); if (n < 2) return NaN;
        double T = st.v(), D = stt.v();
        if (slope_ill(n, T, D)) return slope_direct(0, false, 0, 0.0);
        return (n * stv.v() - T * s1.v()) / (n * D - T * T);
    }
    double ema(size_t i) const { if (i >= emas.size()) throw std::out_of_range("no EMA registered at that index"); return emas[i].value(); }
    i64 last_time_ns() const { return last_t; }

    std::unique_ptr<SimResult> simulate(const double* v, py::ssize_t m, const i64* t, py::ssize_t mt, bool sequential, std::vector<double> qs) const;
};

struct SimResult {   // per hypothetical value: everything needed for every statistic
    py::ssize_t m = 0;
    std::vector<double> cnt, sum, mean, m2, mx, mn, first, last, slope; std::vector<std::vector<double>> ema;
    // lazy quantiles (independent mode): remaining sorted window (shared per expiry group) + the value
    std::vector<std::shared_ptr<const std::vector<double>>> groups; std::vector<int> gid; std::vector<double> val;
    // eager quantiles (sequential mode)
    std::map<double, std::vector<double>> eq; bool lazy = true;
    void resize(py::ssize_t m_, size_t ne) { m = m_; for (auto* a : {&cnt, &sum, &mean, &m2, &mx, &mn, &first, &last, &slope}) a->assign(m, NaN); ema.assign(ne, std::vector<double>(m, NaN)); }
    std::vector<double> quantile(double qq, i64 minp) const {
        if (!(qq >= 0 && qq <= 1)) throw std::invalid_argument("q must be in [0, 1]");
        if (!lazy) {
            auto it = eq.find(qq);
            if (it == eq.end()) throw std::invalid_argument("sequential simulate: pass this q in simulate(..., q=[...]) (median is always available)");
            if (minp <= 1) return it->second;
            std::vector<double> r = it->second; for (py::ssize_t j = 0; j < m; ++j) if (cnt[j] < minp) r[j] = NaN; return r;
        }
        std::vector<double> out(m);
        for (py::ssize_t j = 0; j < m; ++j) {
            const auto& G = *groups[gid[j]]; double x = val[j];
            if (x == x) {
                i64 pos = std::upper_bound(G.begin(), G.end(), x) - G.begin();
                out[j] = quantile_of([&](i64 i) { return i < pos ? G[i] : (i == pos ? x : G[i - 1]); }, (i64)G.size() + 1, qq, minp);
            } else out[j] = quantile_of([&](i64 i) { return G[i]; }, (i64)G.size(), qq, minp);
        }
        return out;
    }
};

inline std::unique_ptr<SimResult> Window::simulate(const double* v, py::ssize_t m, const i64* t, py::ssize_t mt, bool sequential, std::vector<double> qs) const {
    auto R = std::make_unique<SimResult>(); R->resize(m, emas.size());
    auto T = [&](py::ssize_t j) -> i64 { return mt == 0 ? last_t : (mt == 1 ? t[0] : t[j]); };
    for (py::ssize_t j = 0; j < m; ++j) if (has_t && T(j) < last_t) throw std::invalid_argument("simulate: t must be >= the last appended time");
    if (sequential) {
        R->lazy = false; qs.push_back(0.5); for (double qq : qs) R->eq[qq].assign(m, NaN);
        Window W(*this);
        for (py::ssize_t j = 0; j < m; ++j) {
            if (j > 0 && T(j) < T(j - 1)) throw std::invalid_argument("sequential simulate: times must be non-decreasing");
            W.append_ns(T(j), v[j]);
            R->cnt[j] = (double)W.count(); R->sum[j] = W.sum(); R->mean[j] = W.mean(); R->m2[j] = W.m2(); R->mx[j] = W.max(); R->mn[j] = W.min();
            R->first[j] = W.first(); R->last[j] = W.last(); R->slope[j] = W.slope();
            for (size_t e = 0; e < emas.size(); ++e) R->ema[e][j] = W.emas[e].value();
            for (auto& kv : R->eq) kv.second[j] = W.quantile(kv.first, 1);
        }
        return R;
    }
    // ---- independent what-ifs: each value is appended alone to the current state
    const i64 n = (i64)q.size();
    std::vector<i64> K(m);
    i64 kmax = 0;
    for (py::ssize_t j = 0; j < m; ++j) {
        i64 tj = T(j), k = 0;
        if (maxtime > 0) k = std::partition_point(q.begin(), q.end(), [&](const WElem& e) { return tj - e.t >= maxtime; }) - q.begin();
        if (v[j] == v[j] && maxlen > 0 && n - k >= maxlen) k = n - maxlen + 1;
        K[j] = k; kmax = std::max(kmax, k);
    }
    // prefix sums of the expiring front (only as far as needed)
    std::vector<double> P1(kmax + 1, 0), P2(kmax + 1, 0), Pt(kmax + 1, 0), Ptt(kmax + 1, 0), Ptv(kmax + 1, 0);
    { KSum a, b, c, d, e;
      for (i64 i = 0; i < kmax; ++i) { double dv = q[i].v - vref, dt = (double)(q[i].t - tref) * 1e-9; a.add(dv); b.add(dv * dv); c.add(dt); d.add(dt * dt); e.add(dt * dv);
          P1[i + 1] = a.v(); P2[i + 1] = b.v(); Pt[i + 1] = c.v(); Ptt[i + 1] = d.v(); Ptv[i + 1] = e.v(); } }
    // expiry groups for the order statistics
    std::vector<i64> ks(K.begin(), K.end()); std::sort(ks.begin(), ks.end()); ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    std::map<i64, int> gidx;
    { auto cur = std::make_shared<std::vector<double>>(sorted); i64 at = 0;
      for (i64 k : ks) {
          if (k != at) { auto nx = std::make_shared<std::vector<double>>(*cur); for (i64 i = at; i < k; ++i) nx->erase(std::lower_bound(nx->begin(), nx->end(), q[i].v)); cur = nx; at = k; }
          gidx[k] = (int)R->groups.size(); R->groups.push_back(cur);
      } }
    R->gid.resize(m); R->val.assign(v, v + m);
    const double S1 = s1.v(), S2 = s2.v(), ST = st.v(), STT = stt.v(), STV = stv.v();
    for (py::ssize_t j = 0; j < m; ++j) {
        i64 k = K[j], r = n - k, tj = T(j); double x = v[j]; bool ok = x == x;
        R->gid[j] = gidx[k];
        double a = 0, b = 0, c = 0, d = 0, e = 0;
        if (r > 0) { a = S1 - P1[k]; b = S2 - P2[k]; c = ST - Pt[k]; d = STT - Ptt[k]; e = STV - Ptv[k]; }
        double ref = (n > 0) ? vref : (ok ? x : 0.0); i64 tr_ = (n > 0) ? tref : tj;
        if (ok) { double dv = x - ref, dt = (double)(tj - tr_) * 1e-9; a += dv; b += dv * dv; c += dt; d += dt * dt; e += dt * dv; }
        i64 N = r + (ok ? 1 : 0); R->cnt[j] = (double)N;
        i64 run2 = ok ? (x == run_v ? run + 1 : 1) : run;
        if (N > 0) {
            R->sum[j] = ref * (double)N + a; R->mean[j] = ref + a / (double)N; double mm = b - a * a / (double)N; R->m2[j] = (run2 >= N || mm < 0) ? 0.0 : mm;
            if (N >= 2) { double Nd = (double)N; R->slope[j] = slope_ill(Nd, c, d) ? slope_direct(k, ok, tj, x) : (Nd * e - c * a) / (Nd * d - c * c); }
            double hi = NaN, lo = NaN;
            if (r > 0) {
                i64 s0 = q[k].seq;
                hi = std::partition_point(qmax.begin(), qmax.end(), [&](const WElem& w) { return w.seq < s0; })->v;
                lo = std::partition_point(qmin.begin(), qmin.end(), [&](const WElem& w) { return w.seq < s0; })->v;
                R->first[j] = q[k].v; R->last[j] = ok ? x : q.back().v;
            } else { R->first[j] = x; R->last[j] = x; }
            if (ok) { hi = (hi == hi && hi >= x) ? hi : x; lo = (lo == lo && lo <= x) ? lo : x; }
            R->mx[j] = hi; R->mn[j] = lo;
        }
        for (size_t ei = 0; ei < emas.size(); ++ei) { EWM w = emas[ei]; R->ema[ei][j] = w.update_ns(tj, x); }
    }
    return R;
}
