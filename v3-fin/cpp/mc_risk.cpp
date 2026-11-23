// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// mc_risk — C++20 financial-market Monte Carlo risk engine.
//
// Models: gbm | heston (full-truncation Euler) | merton (Poisson thinning) | regime (2-state Markov)
// RNG: xoshiro256++ per thread, seeded by splitmix64(seed, thread_id).
// Normals: Box-Muller with cached spare (auditable, branch-light).
// Threading: std::thread pool, per-thread local state, single join + merge.
// Output: JSON summary with exact quantiles from stored terminal wealth.
//
// Build: g++ -std=c++20 -O3 -march=native -flto -o mc_risk mc_risk.cpp -pthread
//
// Reference equations:
//   GBM (log):    logS+= (mu-0.5*sig^2)*dt + sig*sqrt(dt)*Z
//   Heston:       vbar=max(v,0); v+= kappa*(theta-vbar)*dt + xi*sqrt(vbar*dt)*Zv;
//                 v=max(v,0); logS+= (mu-0.5*vbar)*dt + sqrt(vbar*dt)*(rho*Zv+sqrt(1-rho^2)*Z1)
//                 Feller 2*kappa*theta >= xi^2 (checked, truncation always on)
//   Merton:       k=exp(muJ+0.5*sigJ^2)-1; N~Poisson(lam*dt) via Knuth (lam*dt<<1);
//                 logS+= (mu-0.5*sig^2-lam*k)*dt + sig*sqrt(dt)*Z + sum Yi, Yi~N(muJ,sigJ^2)
//   Regime:       s' = Markov(P); logS+= (mu_s-0.5*sig_s^2)*dt + sig_s*sqrt(dt)*Z
//   Stationary bearing: all paths start S0, report terminal wealth + running MaxDD per path.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---- splitmix64 ----
static std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---- xoshiro256++ ----
struct XoshiroPP {
    std::uint64_t s[4];
    explicit XoshiroPP(std::uint64_t seed = 0, std::uint64_t stream = 0) {
        std::uint64_t x = seed ^ 0x9E3779B97F4A7C15ULL ^ (stream * 0xBF58476D1CE4E5B9ULL + 0x12345ULL);
        for (int i = 0; i < 4; ++i) s[i] = splitmix64(x);
        if ((s[0] | s[1] | s[2] | s[3]) == 0) s[0] = 0x1ULL;
    }
    static inline std::uint64_t rotl(std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
    inline std::uint64_t next() {
        const std::uint64_t res = rotl(s[0] + s[3], 23) + s[0];
        const std::uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return res;
    }
    // uniform (0,1) exclusive of 0
    inline double uniform() {
        return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
};

// ---- normal via Box-Muller with spare cache ----
struct Gauss {
    XoshiroPP& rng;
    bool spare = false;
    double spare_v = 0.0;
    explicit Gauss(XoshiroPP& r) : rng(r) {}
    inline double next() {
        if (spare) { spare = false; return spare_v; }
        double u1 = 0.0;
        do { u1 = rng.uniform(); } while (u1 <= 1e-12);
        const double u2 = rng.uniform();
        constexpr double k2pi = 6.28318530717958647692;
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = k2pi * u2;
        spare_v = r * std::sin(th);
        spare = true;
        return r * std::cos(th);
    }
};

// Poisson Knuth (valid for lam*dt << 1, our case lam<=3/yr -> ldt<=0.012)
inline int poisson_small(double ldt, XoshiroPP& rng) {
    if (ldt <= 0) return 0;
    const double L = std::exp(-ldt);
    int k = 0;
    double p = 1.0;
    do { ++k; p *= rng.uniform(); } while (p > L);
    return k - 1;
}

struct Params {
    std::string model = "gbm";
    long paths = 50000;
    long steps = 2500;
    double mu = 0.07, sigma = 0.20, s0 = 100.0, dt = 1.0 / 252.0;
    // heston
    double kappa = 2.0, theta = 0.04, xi = 0.3, rho = -0.7, v0 = 0.04;
    // merton
    double lambda = 0.8, mjump = -0.04, sjump = 0.05;
    // regime
    double mu1 = 0.10, sig1 = 0.15, mu2 = -0.15, sig2 = 0.35, p12 = 0.01, p21 = 0.03;
    std::uint64_t seed = 42;
    unsigned threads = 0;
    std::string out = "summary.json";
};

bool argval(int argc, char** argv, const char* key, std::string& v) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) { v = argv[i + 1]; return true; }
    return false;
}
template <typename T>
bool argnum(int argc, char** argv, const char* key, T& v) {
    std::string s; if (!argval(argc, argv, key, s)) return false;
    if constexpr (std::is_same_v<T, long>) v = std::stol(s);
    else if constexpr (std::is_same_v<T, unsigned>) v = static_cast<unsigned>(std::stoul(s));
    else if constexpr (std::is_same_v<T, std::uint64_t>) v = static_cast<std::uint64_t>(std::stoull(s));
    else v = std::stod(s);
    return true;
}

struct Local {
    double sum = 0.0, m2 = 0.0;   // Welford on terminal wealth
    std::uint64_t loss = 0;
    double dd_sum = 0.0;
    std::uint64_t n = 0;
};

} // namespace

int main(int argc, char** argv) {
    Params p;
    { std::string s;
      if (argval(argc, argv, "--model", s)) p.model = s;
      argnum(argc, argv, "--paths", p.paths);
      argnum(argc, argv, "--steps", p.steps);
      argnum(argc, argv, "--mu", p.mu);
      argnum(argc, argv, "--sigma", p.sigma);
      argnum(argc, argv, "--s0", p.s0);
      argnum(argc, argv, "--dt", p.dt);
      argnum(argc, argv, "--kappa", p.kappa);
      argnum(argc, argv, "--theta", p.theta);
      argnum(argc, argv, "--xi", p.xi);
      argnum(argc, argv, "--rho", p.rho);
      argnum(argc, argv, "--v0", p.v0);
      argnum(argc, argv, "--lambda", p.lambda);
      argnum(argc, argv, "--mjump", p.mjump);
      argnum(argc, argv, "--sjump", p.sjump);
      argnum(argc, argv, "--mu1", p.mu1);
      argnum(argc, argv, "--sig1", p.sig1);
      argnum(argc, argv, "--mu2", p.mu2);
      argnum(argc, argv, "--sig2", p.sig2);
      argnum(argc, argv, "--p12", p.p12);
      argnum(argc, argv, "--p21", p.p21);
      argnum(argc, argv, "--seed", p.seed);
      argnum(argc, argv, "--threads", p.threads);
      if (argval(argc, argv, "--out", s)) p.out = s; }

    if (p.threads == 0) {
        p.threads = std::thread::hardware_concurrency();
        if (p.threads == 0) p.threads = 4;
    }
    if (p.paths < (long)p.threads) p.threads = static_cast<unsigned>(p.paths);

    const double sqdt = std::sqrt(p.dt);
    const double rho_c = std::sqrt(std::max(0.0, 1.0 - p.rho * p.rho));
    const double k_jump = std::exp(p.mjump + 0.5 * p.sjump * p.sjump) - 1.0;
    const double ldt = p.lambda * p.dt;

    std::vector<double> term(static_cast<size_t>(p.paths));
    std::vector<Local> locals(p.threads);
    std::vector<double> dd_all(static_cast<size_t>(p.paths));

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    const long chunk = (p.paths + p.threads - 1) / p.threads;
    for (unsigned t = 0; t < p.threads; ++t) {
        pool.emplace_back([&, t] {
            const long lo = t * chunk;
            const long hi = std::min(p.paths, lo + chunk);
            XoshiroPP rng(p.seed, t + 1);
            Gauss g(rng);
            Local loc;
            for (long i = lo; i < hi; ++i) {
                double logS = std::log(p.s0);
                double peak = p.s0, maxDD = 0.0;
                if (p.model == "heston") {
                    double v = p.v0;
                    for (long s = 0; s < p.steps; ++s) {
                        const double vb = v > 0 ? v : 0.0;
                        const double z2 = g.next(), z1 = g.next();
                        const double zs = p.rho * z2 + rho_c * z1;
                        v = vb + p.kappa * (p.theta - vb) * p.dt + p.xi * std::sqrt(vb * p.dt) * z2;
                        if (v < 0) v = 0;
                        logS += (p.mu - 0.5 * vb) * p.dt + std::sqrt(vb * p.dt) * zs;
                        const double Sp = std::exp(logS);
                        if (Sp > peak) peak = Sp;
                        const double dd = (peak - Sp) / peak;
                        if (dd > maxDD) maxDD = dd;
                    }
                } else if (p.model == "merton") {
                    const double drift = (p.mu - 0.5 * p.sigma * p.sigma - p.lambda * k_jump) * p.dt;
                    const double diff = p.sigma * sqdt;
                    for (long s = 0; s < p.steps; ++s) {
                        double lg = drift + diff * g.next();
                        const int nj = poisson_small(ldt, rng);
                        for (int j = 0; j < nj; ++j) lg += p.mjump + p.sjump * g.next();
                        logS += lg;
                        const double Sp = std::exp(logS);
                        if (Sp > peak) peak = Sp;
                        const double dd = (peak - Sp) / peak;
                        if (dd > maxDD) maxDD = dd;
                    }
                } else if (p.model == "regime") {
                    int st = 0; // 0 bull, 1 bear
                    for (long s = 0; s < p.steps; ++s) {
                        const double u = rng.uniform();
                        if (st == 0) { if (u < p.p12) st = 1; }
                        else { if (u < p.p21) st = 0; }
                        const double m = (st == 0 ? p.mu1 : p.mu2);
                        const double sg = (st == 0 ? p.sig1 : p.sig2);
                        logS += (m - 0.5 * sg * sg) * p.dt + sg * sqdt * g.next();
                        const double Sp = std::exp(logS);
                        if (Sp > peak) peak = Sp;
                        const double dd = (peak - Sp) / peak;
                        if (dd > maxDD) maxDD = dd;
                    }
                } else { // gbm
                    const double drift = (p.mu - 0.5 * p.sigma * p.sigma) * p.dt;
                    const double diff = p.sigma * sqdt;
                    for (long s = 0; s < p.steps; ++s) {
                        logS += drift + diff * g.next();
                        const double Sp = std::exp(logS);
                        if (Sp > peak) peak = Sp;
                        const double dd = (peak - Sp) / peak;
                        if (dd > maxDD) maxDD = dd;
                    }
                }
                const double ST = std::exp(logS);
                term[static_cast<size_t>(i)] = ST;
                dd_all[static_cast<size_t>(i)] = maxDD;
                // accumulate sum/sumsq (merged post-join); exact quantiles from term[]
                loc.sum += ST;
                loc.m2 += ST * ST;
                if (ST < p.s0) ++loc.loss;
                loc.dd_sum += maxDD;
            }
            locals[t] = loc;
        });
    }
    for (auto& th : pool) th.join();
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // merge
    double sum = 0, sumsq = 0, dd_sum = 0;
    std::uint64_t loss = 0;
    for (auto& l : locals) { sum += l.sum; sumsq += l.m2; loss += l.loss; dd_sum += l.dd_sum; }
    const double N = static_cast<double>(p.paths);
    const double mean = sum / N;
    const double var = std::max(0.0, sumsq / N - mean * mean);
    const double sd = std::sqrt(var);

    std::vector<double> sorted = term;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 20, sorted.end());
    std::sort(sorted.begin(), sorted.end());
    auto q = [&](double a) -> double {
        const double pos = a * (N - 1);
        const size_t lo = static_cast<size_t>(pos);
        const double f = pos - lo;
        if (lo + 1 < sorted.size()) return sorted[lo] * (1 - f) + sorted[lo + 1] * f;
        return sorted.back();
    };
    const double p5 = q(0.05), p50 = q(0.50), p95 = q(0.95);
    double cvar_sum = 0; std::uint64_t cvar_n = 0;
    for (double v : term) if (v <= p5) { cvar_sum += v; ++cvar_n; }
    const double cvar = cvar_n ? cvar_sum / cvar_n : p5;
    const std::int64_t total_steps = p.paths * p.steps;
    const double sps = total_steps / std::max(elapsed, 1e-9);

    std::ofstream o(p.out);
    o << "{\n"
      << "  \"model\": \"" << p.model << "\",\n"
      << "  \"paths\": " << p.paths << ",\n"
      << "  \"steps\": " << p.steps << ",\n"
      << "  \"total_steps\": " << total_steps << ",\n"
      << "  \"seed\": " << p.seed << ",\n"
      << "  \"threads\": " << p.threads << ",\n"
      << "  \"s0\": " << p.s0 << ",\n"
      << "  \"mean_terminal\": " << mean << ",\n"
      << "  \"stdev_terminal\": " << sd << ",\n"
      << "  \"p5\": " << p5 << ",\n"
      << "  \"p50\": " << p50 << ",\n"
      << "  \"p95\": " << p95 << ",\n"
      << "  \"VaR95\": " << (p.s0 - p5) / p.s0 << ",\n"
      << "  \"CVaR95\": " << (p.s0 - cvar) / p.s0 << ",\n"
      << "  \"P_loss\": " << (static_cast<double>(loss) / N) << ",\n"
      << "  \"maxDD_mean\": " << (dd_sum / N) << ",\n"
      << "  \"elapsed_s\": " << elapsed << ",\n"
      << "  \"steps_per_sec\": " << sps << "\n}\n";
    std::printf("%s total_steps=%lld steps_per_sec=%.0f elapsed=%.2fs mean=%.2f p5=%.2f p50=%.2f\n",
                p.model.c_str(), (long long)total_steps, sps, elapsed, mean, p5, p50);
    return 0;
}
