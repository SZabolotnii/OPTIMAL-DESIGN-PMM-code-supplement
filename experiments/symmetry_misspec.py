"""Symmetry misspecification of the per-stratum degree-3 PMM (Proposition prop:asym, Section 6).

The stratify-and-pool PMM_3 imposes symmetry in each stratum (Lemma lem:g3). On a skewed
stratum the odd-power equation is not centred and its root is biased. To first order in the
skewness the population root is t = -sigma*gamma3*gamma4/D with D = 6 + 9*gamma4 + gamma6,
and under local asymmetry gamma3 = h/sqrt(n) the normalised MSE relative to the stratum mean
is g3 + (1 - g3)*h^2/D. D is also the asymptotic variance of sqrt(n)*(sample skewness) under
the symmetric law, so the stratum gains iff |gamma3| is below one asymptotic standard error of
its own sample skewness.

Part 1 checks the bias and the MSE ratio by Monte Carlo on skewed Beta(a, a + d) errors.
Part 2 evaluates, for the KURT design of Section 5 (closed-form field, a(x) = 8 - 7x), the
skewness at which the A-advantage of the PMM_3 design + estimator over the OLS design + OLS
estimator, and the design gain at a fixed PMM_3 estimator, vanish.
Part 3 recomputes the large-N limit of the two-stage design of experiments/two_stage_limit.R
with the field the procedure actually estimates in the limit: the calibrated field at the three
pilot doses, interpolated linearly in x (two_stage.R, line 104), instead of the full field.

    python3 experiments/symmetry_misspec.py > experiments/symmetry_misspec.log
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import betaln

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(20261028)


# ---------------------------------------------------------------- part 1: one stratum
def beta_central(a, b):
    """Mean, sd and central moments m2..m6 of Beta(a, b), exactly."""
    raw = [np.exp(betaln(a + r, b) - betaln(a, b)) for r in range(7)]
    mu = raw[1]
    m = [sum(math.comb(r, j) * raw[j] * (-mu) ** (r - j) for j in range(r + 1)) for r in range(7)]
    return mu, np.sqrt(m[2]), m


def cumulants(m):
    s = m[2]
    g3 = m[3] / s**1.5
    g4 = m[4] / s**2 - 3
    # standardised sixth cumulant of a general law (reduces to the paper's formula when m3 = 0)
    k6 = m[6] - 15 * m[4] * m[2] - 10 * m[3] ** 2 + 30 * m[2] ** 3
    return g3, g4, k6 / s**3


def pmm3_root(y):
    """Lemma lem:g3(c): kappa from central moments about ybar, root nearest ybar (rows of y)."""
    ybar = y.mean(1, keepdims=True)
    e = y - ybar
    m2, m4, m6 = (e**2).mean(1), (e**4).mean(1), (e**6).mean(1)
    kap = (m6 - 3 * m2 * m4) / (m4 - 3 * m2**2)
    # sum_i psi(y_i - mu) with psi(t) = t*(kap - t^2), as a cubic in s = mu - ybar:
    # sum (e - s)(kap - (e - s)^2) = n[ -kap s - (m3 - 3 m2 s - s^3) ]   (sum e = 0)
    m3 = (e**3).mean(1)
    out = np.empty(len(y))
    for i in range(len(y)):
        r = np.roots([1.0, 0.0, 3 * m2[i] - kap[i], -m3[i]])
        r = r[np.abs(r.imag) < 1e-9].real
        out[i] = ybar[i, 0] + r[np.argmin(np.abs(r))]
    return out


print("== part 1: per-stratum PMM_3 with symmetry imposed, on skewed Beta(a, a+d) errors ==")
print(" a     d      n   gamma3   z=sqrt(n)g3/sqrt(D)  bias(pop.root)  bias(1st ord)  bias(MC) [se]"
      "    nMSE/var: theory   MC [se]")
M = 20000
for a in (1.0, 2.0):
    for d in (0.0, 0.05, 0.15):
        mu, sd, m = beta_central(a, a + d)
        g3s, g4, g6 = cumulants(m)
        mstd = [m[r] / sd**r for r in range(7)]
        kap = (mstd[6] - 3 * mstd[4]) / (mstd[4] - 3)
        r = np.roots([1.0, 0.0, 3 - kap, -mstd[3]])
        r = r[np.abs(r.imag) < 1e-9].real
        tpop = r[np.argmin(np.abs(r))]
        D = 6 + 9 * g4 + g6
        gg = 1 - g4**2 / D
        for n in (400, 2000):
            y = (rng.beta(a, a + d, size=(M, n)) - mu) / sd
            est = pmm3_root(y)
            bias, se_b = est.mean(), est.std() / np.sqrt(M)
            nmse = n * (est**2)
            th = gg + n * tpop**2
            z = np.sqrt(n) * g3s / np.sqrt(D)
            print(f"{a:3.1f} {d:5.2f} {n:6d}  {g3s:7.4f}  {z:8.3f}            {tpop:9.5f}      "
                  f"{-g3s * g4 / D:9.5f}   {bias:9.5f} [{se_b:.5f}]     {th:7.4f}   {nmse.mean():7.4f} "
                  f"[{nmse.std() / np.sqrt(M):.4f}]")
print("theory column: g3 + n*t^2 with g3 at the perturbed law's even cumulants and t the population root")
# D = 6 + 9 g4 + g6 is the asymptotic variance of sqrt(n) * (sample skewness) under a symmetric law
for a in (1.0, 2.0):
    mu, sd, m = beta_central(a, a)
    _, g4, g6 = cumulants(m)
    e = rng.beta(a, a, size=(M, 2000))
    e = e - e.mean(1, keepdims=True)
    b1 = (e**3).mean(1) / (e**2).mean(1) ** 1.5
    print(f"Beta({a:.0f},{a:.0f}): D = {6 + 9 * g4 + g6:.4f}, n*Var(sample skewness) at n = 2000: {2000 * b1.var():.4f}")

# ---------------------------------------------------------------- part 2: the KURT design
GRID = np.round(np.arange(0, 1.0000001, 0.02), 10)
FX = np.column_stack([np.ones_like(GRID), GRID, GRID**2])


def aopt(H, it=20000, tol=1e-12):
    w = np.full(len(H), 1 / len(H))
    for _ in range(it):
        Mi = np.linalg.inv((H.T * w) @ H)
        dA = np.einsum("ij,jk,ik->i", H, Mi @ Mi, H)
        wn = w * np.sqrt(np.maximum(dA, 0))
        wn /= wn.sum()
        if np.max(np.abs(wn - w)) < tol:
            return wn
        w = wn
    return w


def closed(a):
    u = 2 * a + 3
    g4, g6 = -6 / u, 240 / (u * (u + 2))
    D = 6 + 9 * g4 + g6
    return 1 - g4**2 / D, g4, D


gC, g4C, DC = closed(8 - 7 * GRID)
wO, wP = aopt(FX), aopt(FX / np.sqrt(gC)[:, None])
supp = lambda w: "{" + ", ".join(f"{x:.2f}:{v:.3f}" for x, v in zip(GRID, w) if v > 1e-3) + "}"
print("\n== part 2: KURT design (closed-form field), A-criterion, constant skewness gamma3(x) = gamma ==")
print("OLS-optimal  ", supp(wO))
print("PMM3-optimal ", supp(wP))


def trM(w, g):
    return np.trace(np.linalg.inv((FX.T * (w / g)) @ FX))


def bias_vec(w, g, c):
    """sqrt(N)*bias / (sigma*h) of the pooled estimator, h = sqrt(N)*gamma, stratum bias -c."""
    W = w / g
    return np.linalg.solve((FX.T * W) @ FX, FX.T @ (W * -c))


c = g4C / DC                                   # first-order per-unit-skewness bias coefficient
tO = trM(wO, np.ones_like(GRID))               # OLS design, OLS estimator (unbiased)
tPP, tOP = trM(wP, gC), trM(wO, gC)             # PMM3 estimator on the PMM3 / OLS design
bPP, bOP = bias_vec(wP, gC, c), bias_vec(wO, gC, c)
print(f"A-gain, PMM3 design under the field (gamma = 0): {tOP / tPP:.4f}")
print(f"A-gain of PMM3 design+estimator over OLS design+OLS (gamma = 0): {tO / tPP:.4f}")
hstar_full = np.sqrt((tO - tPP) / (bPP @ bPP))
print(f"break-even h = sqrt(N)*gamma, PMM3 design+estimator vs OLS design+OLS: {hstar_full:.3f}")
# design gain at a fixed PMM3 estimator, both designs biased
BP, BO = bPP @ bPP, bOP @ bOP
cross = np.sqrt((tOP - tPP) / (BP - BO)) if BP > BO else np.inf
print(f"design gain at a fixed PMM3 estimator stays >= 1 up to h = {cross:.3f}; limit h -> inf {BO / BP:.4f} "
      f"(||b_PMM3 design|| = {np.sqrt(bPP @ bPP):.3f}, ||b_OLS design|| = {np.sqrt(bOP @ bOP):.3f})")
for N in (600, 1200, 2400):
    print(f"  N = {N:5d}: break-even |gamma| full comparison = {hstar_full / np.sqrt(N):.4f}, "
          f"design-gain comparison = {cross / np.sqrt(N):.4f}")
lo, hi = np.sqrt(DC.min()), np.sqrt(DC.max())
print(f"sqrt(D) over the KURT field: {lo:.3f} (a = 1) to {hi:.3f} (a = 8); stratum break-even |gamma3| = sqrt(D/n_k)")

# ---------------------------------------------------------------- part 3: two-stage limit
cal = json.load(open(ROOT / "results/summary.json"))["kurt"]["calib"]
pa = np.array([r["param"] for r in cal])[::-1]
pg = np.minimum(np.array([r["g3"] for r in cal])[::-1], 1)
g = np.minimum(np.interp(8 - 7 * GRID, pa, pg), 1)
wO, wP = aopt(FX), aopt(FX / np.sqrt(g)[:, None])
pilot = np.array([0.0, 0.5, 1.0])
gint = np.interp(GRID, pilot, np.interp(8 - 7 * pilot, pa, np.minimum(pg, 1)))
wI = aopt(FX / np.sqrt(gint)[:, None])
print("\n== part 3: two-stage limit (calibrated field) ==")
print(f"oracle one-stage A-gain {trM(wO, g) / trM(wP, g):.4f}; design for the interpolated field {supp(wI)}")
print(f"one-stage design built on the interpolated field, A-gain under the true field: "
      f"{trM(wO, g) / trM(wI, g):.4f}")
for r in (1 / 3, 0.2, 0.1, 0.05, 0.0):
    for lab, tgt in (("full field", wP), ("interpolated", wI)):
        c1 = r * wO
        de = np.maximum(tgt - c1, 0)
        w = c1 + de / de.sum() * (1 - r)
        print(f"pilot fraction {r:.3f}, {lab:12s}: limit A-gain {trM(wO, g) / trM(w, g):.4f}")
