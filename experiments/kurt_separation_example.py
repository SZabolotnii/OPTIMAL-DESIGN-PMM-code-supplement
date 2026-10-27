"""Numbers of Example ex:kurt (a numerical instance of separation, Section 3).

The OLS A-optimal design for the quadratic mean f(x) = (1, x, x^2) on [0, 1] is
supported on {0, t*, 1}, with weights proportional to the column norms of F^{-1}
(F the 3x3 matrix of rows f(x_k)^T); t* minimises the A-criterion over the interior
node (t = 1/2 itself is not exactly optimal). The script checks the OLS
A-equivalence condition for this design on a fine grid of [0, 1], then evaluates
the A-equivalence condition for M_g under the closed-form KURT field g_3(x) of
Appendix B (symmetric Beta(a, a) errors, a(x) = 8 - 7x) and shows that it fails,
so the design is not A-optimal for M_g. The paper quotes t* = 0.4998, the weights
(0.322, 0.486, 0.192), max h'M_g^-2 h = 131.8 at x = 0 and tr M_g^-1 = 110.4.

    python3 experiments/kurt_separation_example.py > experiments/kurt_separation_example.log
"""
import numpy as np
from scipy.optimize import minimize_scalar


def f(x):
    return np.array([1.0, x, x * x])


def a_design(t):
    F = np.array([f(x) for x in (0.0, t, 1.0)])
    w = np.linalg.norm(np.linalg.inv(F), axis=0)
    return F, w / w.sum()


def g3(x):
    # closed-form third-order reduction for Beta(a, a) errors, eq. (g3) of Appendix B
    u = 2 * (8 - 7 * x) + 3
    return 1 - 6 * (u + 2) / (u * (u**2 - 7 * u + 22))


crit = lambda t: np.trace(np.linalg.inv((lambda F, w: (F.T * w) @ F)(*a_design(t))))
tstar = minimize_scalar(crit, bounds=(0.3, 0.7), method="bounded", options={"xatol": 1e-12}).x
F, w = a_design(tstar)
xx = np.linspace(0.0, 1.0, 200001)
H = np.column_stack([np.ones_like(xx), xx, xx**2])

Mi = np.linalg.inv((F.T * w) @ F)
phi = np.einsum("ij,jk,ik->i", H, Mi @ Mi, H)
print(f"t*      = {tstar:.5f}")
print("weights =", np.round(w, 4))
print(f"OLS:  max f'M^-2 f = {phi.max():.6f}  tr M^-1 = {np.trace(Mi):.6f}  "
      f"condition holds: {bool(phi.max() <= np.trace(Mi) * (1 + 1e-10))}")

gk = g3(np.array([0.0, tstar, 1.0]))
Mgi = np.linalg.inv((F.T * (w / gk)) @ F)
Hh = H / np.sqrt(g3(xx))[:, None]
phig = np.einsum("ij,jk,ik->i", Hh, Mgi @ Mgi, Hh)
print("g_3 on the support =", np.round(gk, 4))
print(f"PMM3: max h'M_g^-2 h = {phig.max():.3f} at x = {xx[phig.argmax()]:.3f}  tr M_g^-1 = {np.trace(Mgi):.3f}  "
      f"condition violated: {bool(phig.max() > np.trace(Mgi) * 1.001)}")
