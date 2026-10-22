"""Symbolic and numerical checks for Appendix A (Proposition 1) and Appendix B (g_3).

Run from the repository root:  python3 experiments/g3_derivation.py
(needs sympy, numpy, scipy). Writes
  experiments/g3_beta_closed_form.csv   closed-form g_3(a) for Beta(a,a) vs the n=400 calibration
  experiments/prop1_mc.csv              Monte Carlo check of Proposition 1 with per-stratum PMM_3
and prints PASS/FAIL for every check; the log is experiments/g3_derivation.log.

Checks
  B1  the asymptotic variance of the degree-3 estimating equation psi_k(e)=e(k-e^2),
      minimised over k, is sigma^2 g_3 with g_3 = 1 - gamma4^2/(6+9 gamma4+gamma6);
      the minimiser is EstemPMM's kappa = (m6-3 m2 m4)/(m4-3 m2^2); k -> infinity gives 1.
  B2  6+9 gamma4+gamma6 = E[He3(eps)^2] and gamma4 = E[eps He3(eps)] for a standardised
      symmetric eps, so g_3 = 1 - corr^2(eps, He3(eps)).
  B3  the plug-in of kappa has no first-order effect: d/dk E[psi_k(e)] = E[e] = 0.
  B4  closed-form standardised cumulants of the centred Beta(a,a) agree with scipy.
  A1  Proposition 1 algebra: (F'WF)^{-1}F'W W^{-1} W F (F'WF)^{-1} = (F'WF)^{-1} with
      W = diag(w_k/g_k), and F'WF = M_g.
  A2  Monte Carlo: per-stratum PMM_3 location on Beta(a_k,a_k) errors, GLS pooling with
      weights n_k/g_3(a_k); N * Cov(theta_hat) is compared with sigma^2 M_g^{-1}.
"""
import csv
import math
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import stats

HERE = Path(__file__).resolve().parent
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- B1
k, m2, m4, m6 = sp.symbols("kappa m2 m4 m6", real=True)
g4, g6 = sp.symbols("gamma4 gamma6", real=True)
# psi_k(e) = e (k - e^2); E psi^2 and E d psi / d mu for a symmetric law (m1 = m3 = m5 = 0)
E_psi2 = k**2 * m2 - 2 * k * m4 + m6
E_dpsi = 3 * m2 - k                      # d/dmu of (y-mu)(k-(y-mu)^2) = -(k - 3e^2)
avar = E_psi2 / E_dpsi**2                # n * Var(mu_hat) (sandwich)
ratio = sp.simplify(avar / m2)           # reduction relative to the mean
k_star = sp.solve(sp.diff(ratio, k), k)
kappa_estem = (m6 - 3 * m2 * m4) / (m4 - 3 * m2**2)
check("B1a minimiser equals EstemPMM kappa", any(sp.simplify(ks - kappa_estem) == 0 for ks in k_star),
      f"roots={k_star}")
g3_at_star = sp.simplify(ratio.subs(k, kappa_estem))
subs_std = {m4: (3 + g4) * m2**2, m6: (g6 + 15 * g4 + 15) * m2**3}
g3_closed = 1 - g4**2 / (6 + 9 * g4 + g6)
check("B1b g3(kappa*) = 1 - g4^2/(6+9g4+g6)",
      sp.simplify(g3_at_star.subs(subs_std) - g3_closed) == 0)
check("B1c kappa -> infinity recovers the mean (ratio -> 1)", sp.limit(ratio, k, sp.oo) == 1)
# second-order condition: the stationary point is a minimum whenever the denominator is positive
d2 = sp.simplify(sp.diff(ratio, k, 2).subs(k, kappa_estem).subs(subs_std).subs(m2, 1))
print("    d2 ratio/dk2 at kappa* (sigma=1):", sp.factor(d2))
check("B1d d2 r/dk2 at kappa* = 2 g4^4/(6+9g4+g6)^3", sp.simplify(d2 - 2 * g4**4 / (6 + 9 * g4 + g6)**3) == 0)
check("B1e kappa* - 3 m2 = m2 (6+9g4+g6)/g4",
      sp.simplify((kappa_estem - 3 * m2).subs(subs_std) - m2 * (6 + 9 * g4 + g6) / g4) == 0)
check("B1f exact expansion psi(e-t) = psi(e) - t(k-3e^2) - 3e t^2 + t^3",
      sp.expand((sp.Symbol('e') - sp.Symbol('t')) * (k - (sp.Symbol('e') - sp.Symbol('t'))**2)
                - (sp.Symbol('e') * (k - sp.Symbol('e')**2) - sp.Symbol('t') * (k - 3 * sp.Symbol('e')**2)
                   - 3 * sp.Symbol('e') * sp.Symbol('t')**2 + sp.Symbol('t')**3)) == 0)

# ---------------------------------------------------------------- B2
e = sp.symbols("e", real=True)
He3 = e**3 - 3 * e
# moments of a standardised symmetric law: E e^2=1, E e^4=3+g4, E e^6=g6+15 g4+15, odd = 0
mom = {2: 1, 4: 3 + g4, 6: g6 + 15 * g4 + 15}


def E(poly):
    poly = sp.Poly(sp.expand(poly), e)
    tot = 0
    for (p,), c in poly.terms():
        tot += c * (1 if p == 0 else mom.get(p, 0))
    return sp.expand(tot)


check("B2a E[He3^2] = 6+9g4+g6", sp.simplify(E(He3**2) - (6 + 9 * g4 + g6)) == 0)
check("B2b E[e He3] = g4", sp.simplify(E(e * He3) - g4) == 0)

# ---------------------------------------------------------------- B3
check("B3 d/dkappa E[psi] = E[e] = 0 (symmetric law)", E(sp.diff(e * (k - e**2), k)) == 0)


# ---------------------------------------------------------------- B4  Beta(a,a)
def beta_std_cumulants(a):
    """gamma4, gamma6 of a centred Beta(a,a), from E(X-1/2)^{2j} = 4^{-j} (1/2)_j/(a+1/2)_j."""
    def cm(j):
        return 4.0**-j * math.prod((0.5 + i) / (a + 0.5 + i) for i in range(j))
    mu2, mu4, mu6 = cm(1), cm(2), cm(3)
    g4_ = mu4 / mu2**2 - 3
    g6_ = mu6 / mu2**3 - 15 * mu4 / mu2**2 + 30
    return g4_, g6_


def g3_of(g4_, g6_):
    return 1 - g4_**2 / (6 + 9 * g4_ + g6_)


from scipy import integrate

maxerr = 0.0
for a in (8, 5, 3.5, 2.5, 2, 1.6, 1.3, 1.0):
    X = stats.beta(a, a)
    # central moments by quadrature (scipy's raw Beta moments carry ~1e-13 noise that the
    # raw-to-central conversion amplifies; quadrature of (x-1/2)^r is exact to ~1e-15)
    m = [integrate.quad(lambda x, r=r: (x - 0.5)**r * X.pdf(x), 0, 1, epsabs=1e-15, epsrel=1e-13)[0]
         for r in (2, 4, 6)]
    g4n = m[1] / m[0]**2 - 3
    g6n = m[2] / m[0]**3 - 15 * m[1] / m[0]**2 + 30
    g4c, g6c = beta_std_cumulants(a)
    maxerr = max(maxerr, abs(g4n - g4c), abs(g6n - g6c))
check("B4 Beta(a,a) closed-form cumulants match quadrature", maxerr < 1e-9, f"max abs err {maxerr:.2e}")
check("B4b uniform: g4=-6/5, g6=48/7, g3=0.300",
      abs(beta_std_cumulants(1)[0] + 1.2) < 1e-12 and abs(beta_std_cumulants(1)[1] - 48 / 7) < 1e-12
      and abs(g3_of(*beta_std_cumulants(1)) - 0.3) < 1e-12)

# closed forms in a: gamma4 = -6/u, gamma6 = 240/(u(u+2)), g3 = 1 - 6(u+2)/(u(u^2-7u+22)), u = 2a+3
asym, usym = sp.symbols("a u", positive=True)
cmom = lambda j: sp.Rational(1, 4**j) * sp.prod([(sp.Rational(1, 2) + i) / (asym + sp.Rational(1, 2) + i) for i in range(j)])
g4a = sp.simplify(cmom(2) / cmom(1)**2 - 3)
g6a = sp.simplify(cmom(3) / cmom(1)**3 - 15 * cmom(2) / cmom(1)**2 + 30)
uu = 2 * asym + 3
check("B5a Beta gamma4 = -6/(2a+3)", sp.simplify(g4a + 6 / uu) == 0)
check("B5b Beta gamma6 = 240/((2a+3)(2a+5))", sp.simplify(g6a - 240 / (uu * (uu + 2))) == 0)
check("B5c Beta g3 = 1 - 6(u+2)/(u(u^2-7u+22)), u=2a+3",
      sp.simplify(g3_closed.subs({g4: g4a, g6: g6a}) - (1 - 6 * (uu + 2) / (uu * (uu**2 - 7 * uu + 22)))) == 0)

hu = 6 * (usym + 2) / (usym * (usym**2 - 7 * usym + 22))
dh = sp.factor(sp.diff(hu, usym))
check("B5d g3(a) strictly increasing in a > 0 (dh/du < 0 for u = 2a+3 > 3)",
      sp.simplify(dh - (-6 * (usym - 2) * (2 * usym**2 + 3 * usym - 22) / (usym**2 * (usym**2 - 7 * usym + 22)**2))) == 0
      and max(sp.solve(2 * usym**2 + 3 * usym - 22, usym)) < 3, f"dh/du = {dh}")

# calibrated values (n = 400, EstemPMM lm_pmm3, M_cal = 2500) from results/summary.json
import json
cal = json.loads((HERE.parent / "results" / "summary.json").read_text())["kurt"]["calib"]
rows = []
for r in cal:
    a = float(r["param"])
    g4_, g6_ = beta_std_cumulants(a)
    rows.append({"a": a, "gamma4": round(g4_, 6), "gamma6": round(g6_, 6),
                 "g3_closed_form": round(g3_of(g4_, g6_), 6), "g3_calibrated_n400": r["g3"]})
with open(HERE / "g3_beta_closed_form.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(HERE.parent / "tables" / "tab_g3_closed.tex").write_text("\n".join(
    ["\\begin{tabular}{rrrrr}", "\\toprule",
     "Beta shape $a$ & $\\gamma_4$ & $\\gamma_6$ & $g_3$ closed form & $\\hat g_3$ calibrated ($n=400$) \\\\", "\\midrule"]
    + [f"{r['a']:.1f} & ${r['gamma4']:.3f}$ & {r['gamma6']:.3f} & {r['g3_closed_form']:.3f} & {r['g3_calibrated_n400']:.3f} \\\\" for r in rows]
    + ["\\bottomrule", "\\end{tabular}", ""]))
print("    a     gamma4    gamma6   g3_closed  g3_cal(n=400)")
for r in rows:
    print(f"    {r['a']:<4}  {r['gamma4']:8.4f}  {r['gamma6']:8.4f}  {r['g3_closed_form']:8.4f}  {r['g3_calibrated_n400']:8.4f}")

# ---------------------------------------------------------------- A1
K, p = 4, 3
ws = sp.symbols("w1:5", positive=True)
gs = sp.symbols("g1:5", positive=True)
xs = [sp.Rational(0), sp.Rational(3, 10), sp.Rational(7, 10), sp.Rational(1)]
F = sp.Matrix([[1, x, x**2] for x in xs])
W = sp.diag(*[ws[i] / gs[i] for i in range(K)])
Mg = sum(((ws[i] / gs[i]) * F[i, :].T * F[i, :] for i in range(K)), sp.zeros(p, p))
A = (F.T * W * F).inv() * F.T * W
lhs = sp.simplify(A * W.inv() * A.T - (F.T * W * F).inv())
check("A1a sandwich A W^-1 A' = (F'WF)^-1", lhs == sp.zeros(p, p))
check("A1b F'WF = M_g", sp.simplify(F.T * W * F - Mg) == sp.zeros(p, p))


# ---------------------------------------------------------------- A2  Monte Carlo
def pmm3_location(Y):
    """EstemPMM's lm_pmm3 for an intercept-only model, vectorised over rows of Y.
    kappa from the OLS (= mean) residual moments; Newton on sum e (kappa - e^2) = 0."""
    mu = Y.mean(axis=1, keepdims=True)
    r = Y - mu
    M2, M4, M6 = (r**2).mean(1, keepdims=True), (r**4).mean(1, keepdims=True), (r**6).mean(1, keepdims=True)
    kap = (M6 - 3 * M4 * M2) / (M4 - 3 * M2**2)
    for _ in range(100):
        eps = Y - mu
        Z = (eps * (kap - eps**2)).sum(1, keepdims=True)
        J = (3 * eps**2 - kap).sum(1, keepdims=True)
        step = Z / J
        step = np.clip(step, -5, 5)
        mu = mu - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return mu[:, 0]


rng = np.random.default_rng(20261021)
theta = np.array([2.0, 1.5, -1.0])
xk = np.array([0.0, 0.3, 0.7, 1.0])
ak = np.array([8.0, 3.0, 1.5, 1.0])          # Beta(a,a) shape per stratum
wk = np.array([0.30, 0.25, 0.25, 0.20])
N, Mrep = 8000, 4000
nk = np.round(N * wk).astype(int)
Fk = np.column_stack([np.ones(4), xk, xk**2])
gk = np.array([g3_of(*beta_std_cumulants(a)) for a in ak])
Mg_num = (Fk.T * (wk / gk)) @ Fk
target = np.linalg.inv(Mg_num)                 # sigma^2 = 1
Wn = np.diag(nk / gk)
Agls = np.linalg.solve(Fk.T @ Wn @ Fk, Fk.T @ Wn)
est = np.empty((Mrep, 3))
mu_hat = np.empty((Mrep, 4))
for j in range(4):
    a = ak[j]
    Y = theta @ Fk[j] + (rng.beta(a, a, size=(Mrep, nk[j])) - 0.5) * 2 * math.sqrt(2 * a + 1)
    mu_hat[:, j] = pmm3_location(Y)
est = mu_hat @ Agls.T
covN = N * np.cov(est.T)
# per-stratum reduction actually realised at these n_k (finite-sample)
g_real = np.array([nk[j] * mu_hat[:, j].var(ddof=1) for j in range(4)])
rel = np.abs(np.diag(covN) / np.diag(target) - 1)
print("    g3 closed form per stratum:", np.round(gk, 4), " realised n_k*Var:", np.round(g_real, 4))
print("    diag N*Cov(theta_hat):", np.round(np.diag(covN), 4), " target diag sigma^2 M_g^-1:", np.round(np.diag(target), 4))
# Monte Carlo SE of a variance estimate ~ sqrt(2/M) relative -> 2.2% at M=4000; allow 3 SE plus finite-n bias
check("A2 N*Cov(theta_hat) matches sigma^2 M_g^{-1} on the diagonal (rel. err < 8%)", rel.max() < 0.08,
      f"max rel err {rel.max():.3f}")
with open(HERE / "prop1_mc.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["quantity", "index", "monte_carlo", "asymptotic"])
    for j in range(4):
        w.writerow(["stratum_reduction", j, f"{g_real[j]:.6f}", f"{gk[j]:.6f}"])
    for i in range(3):
        for jj in range(3):
            w.writerow([f"NCov[{i},{jj}]", "", f"{covN[i, jj]:.6f}", f"{target[i, jj]:.6f}"])
    w.writerow(["meta", "N_M_seed", f"{N}_{Mrep}_20261021", ""])

print("ALL PASS" if ok_all else "SOME CHECK FAILED")
