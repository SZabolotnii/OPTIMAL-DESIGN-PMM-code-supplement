"""Counterexample behind Remark rem:five (added 2026-10-21).

Theorem 2(b) originally claimed that a non-constant reduction field never leaves
M_g(xi) proportional to M(xi). For the quadratic mean f(x) = (1, x, x^2) the
matrices f f^T are Hankel in the moments 1..x^4, a five-dimensional space, so on
six or more support points they are linearly dependent. This script builds a
non-constant field g in (0, 1] on six equally spaced points with equal weights for
which M_g = c M exactly, and prints c (quoted in the paper as 4.78).

    python3 experiments/thm2b_six_point.py
"""
import numpy as np

x = np.linspace(0.0, 1.0, 6)
w = np.full(6, 1 / 6)
V = np.vstack([x**j for j in range(5)])      # sum_k a_k x_k^j = 0 for j = 0..4
a = np.linalg.svd(V)[2][-1]                  # one-dimensional null space
c = 1 + np.abs(a / w).max()                  # makes every 1/g_k >= 1
g = 1 / (c + a / w)


def f(t):
    return np.array([1.0, t, t * t])


M = sum(wk * np.outer(f(t), f(t)) for wk, t in zip(w, x))
Mg = sum(wk / gk * np.outer(f(t), f(t)) for wk, gk, t in zip(w, g, x))
print("support  =", np.round(x, 2))
print("g        =", np.round(g, 4))
print("g in (0,1]:", bool(np.all((g > 0) & (g <= 1))), "| non-constant:", bool(np.ptp(g) > 1e-6))
print(f"c        = {c:.4f}")
print(f"max|Mg - c M| = {np.abs(Mg - c * M).max():.2e}")
