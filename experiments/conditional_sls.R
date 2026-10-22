#!/usr/bin/env Rscript
# =====================================================================
# Second-order design under the GLOBAL conditional SLS estimator.
#
# The locally-adaptive PMM_2 of run_all.R co-estimates a scale in every stratum,
# so its per-stratum location reduction is 1 (Lemma loc2) and its design never
# moves. The model of the paper, however, has ONE common error variance. The
# Wang-Leblanc SLS with conditional weights W_i = E[rho_i rho_i' | x_i]^{-1} and a
# common sigma^2 uses that, and by the T_i cancellation of the SLS = PMM_2 identity
# (companion paper, Theorem 1 and Proposition 1 with the conditional body F_2(x))
# its slope score is f(x) * (F_2(x)^{-1} m(e))_1 with m(e) = (e, e^2 - sigma^2)'.
# The (beta, sigma^2) information is additive over design points:
#   J(x) = [[a(x) f f', c(x) f], [c(x) f', d(x)]],
#   F_2(x)^{-1} = [[a, c], [c, d]] = [[2+g4, -g3], [-g3, 1]] / (2 + g4 - g3^2)   (sigma = 1),
# and the information for beta is the Schur complement after profiling sigma^2.
# A-optimality for beta is an L-criterion on J (multiplicative algorithm, exponent 1/2);
# D-optimality for beta is D_s-optimality with sigma^2 as nuisance.
#
# Cumulant fields are the closed forms of the two error families of run_all.R:
#   centred Gamma(k), k = 4/g3^2:  gamma3 = g3, gamma4 = 1.5 g3^2;
#   symmetric Beta(a, a):          gamma3 = 0,  gamma4 = -6/(2a + 3).
# No EstemPMM call: the design part is deterministic. A Monte Carlo with the
# oracle conditional weights checks that N * Cov(theta_hat) matches J^{-1}.
#
# Usage: Rscript conditional_sls.R [M] [N]      (defaults M=4000, N=2000)
# Output: results/conditional_sls.json, tables/tab_conditional_sls.tex
# =====================================================================
suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
M    <- if (length(args) >= 1) as.integer(args[1]) else 4000L
NMC  <- if (length(args) >= 2) as.integer(args[2]) else 2000L
SEED <- 20261019L
BETA <- c(2.0, 1.5, -1.0)
GRID <- seq(0, 1, by = 0.02)
fX   <- function(x) cbind(1, x, x^2)
P    <- 3L
.argv <- commandArgs(FALSE); .f <- sub("^--file=", "", grep("^--file=", .argv, value = TRUE))
HERE <- if (length(.f)) normalizePath(file.path(dirname(.f), "..")) else "."
cat(sprintf("== conditional_sls | M=%d N=%d | seed=%d ==\n", M, NMC, SEED))

# ---------- cumulant fields (same regimes as run_all.R) ----------
fields <- list(
  KURT_HET = list(g3 = function(x) 0 * x, g4 = function(x) -6 / (2 * (8 - 7 * x) + 3)),
  KURT_HOM = list(g3 = function(x) 0 * x, g4 = function(x) -6 / (2 * 2.0 + 3) + 0 * x),
  SKEW_HET = list(g3 = function(x) 0.2 + 1.6 * x, g4 = function(x) 1.5 * (0.2 + 1.6 * x)^2),
  SKEW_HOM = list(g3 = function(x) 1.0 + 0 * x, g4 = function(x) 1.5 + 0 * x))

# ---------- per-point information blocks ----------
blocks <- function(g3, g4) {
  D <- 2 + g4 - g3^2
  list(a = (2 + g4) / D, c = -g3 / D, d = 1 / D)
}
Jsum <- function(w, x, bl) {
  Fm <- fX(x); J <- matrix(0, P + 1, P + 1)
  J[1:P, 1:P] <- crossprod(Fm * sqrt(w * bl$a))
  J[1:P, P + 1] <- J[P + 1, 1:P] <- colSums(Fm * (w * bl$c))
  J[P + 1, P + 1] <- sum(w * bl$d)
  J
}
Jpts <- function(x, bl) {           # list of (P+1)x(P+1) per-point blocks
  Fm <- fX(x)
  lapply(seq_along(x), function(k) {
    J <- matrix(0, P + 1, P + 1)
    J[1:P, 1:P] <- bl$a[k] * tcrossprod(Fm[k, ])
    J[1:P, P + 1] <- J[P + 1, 1:P] <- bl$c[k] * Fm[k, ]
    J[P + 1, P + 1] <- bl$d[k]
    J
  })
}
Lsel <- diag(c(rep(1, P), 0))
A_val <- function(w, bl) sum(diag(Lsel %*% solve(Jsum(w, GRID, bl))))
D_val <- function(w, bl) { J <- Jsum(w, GRID, bl); as.numeric(determinant(J)$modulus) - log(J[P + 1, P + 1]) }

aopt_sls <- function(bl, it = 20000, tol = 1e-12) {
  Jk <- Jpts(GRID, bl); w <- rep(1 / length(GRID), length(GRID))
  for (i in seq_len(it)) {
    Ji <- solve(Jsum(w, GRID, bl)); G <- Ji %*% Lsel %*% Ji
    dk <- vapply(Jk, function(J) sum(G * J), 0)
    wn <- w * sqrt(pmax(dk, 0) / sum(w * dk)); wn <- wn / sum(wn)
    if (max(abs(wn - w)) < tol) { w <- wn; break }; w <- wn
  }
  w
}
dopt_sls <- function(bl, it = 20000, tol = 1e-12) {
  Jk <- Jpts(GRID, bl); w <- rep(1 / length(GRID), length(GRID))
  for (i in seq_len(it)) {
    J <- Jsum(w, GRID, bl); Ji <- solve(J)
    dk <- vapply(Jk, function(Q) sum(Ji * Q), 0) - bl$d / J[P + 1, P + 1]
    wn <- w * dk / P; wn <- wn / sum(wn)
    if (max(abs(wn - w)) < tol) { w <- wn; break }; w <- wn
  }
  w
}
sep  <- function(w1, w2) 0.5 * sum(abs(w1 - w2))
summ <- function(w, thr = 1e-3) { k <- which(w > thr); data.frame(x = GRID[k], w = round(w[k], 3)) }

# OLS reference: a = 1, c = 0 (any symmetric law gives exactly this; checked in KURT rows)
bl_ols <- list(a = rep(1, length(GRID)), c = rep(0, length(GRID)), d = rep(1 / 2, length(GRID)))
A_ols <- aopt_sls(bl_ols); D_ols <- dopt_sls(bl_ols)
cat("OLS A-optimal:"); print(summ(A_ols), row.names = FALSE)

out <- list()
for (nm in names(fields)) {
  fl <- fields[[nm]]; bl <- blocks(fl$g3(GRID), fl$g4(GRID))
  wA <- aopt_sls(bl); wD <- dopt_sls(bl)
  out[[nm]] <- list(
    sepA = sep(wA, A_ols), gainA = A_val(A_ols, bl) / A_val(wA, bl),
    sepD = sep(wD, D_ols), gainD = exp((D_val(wD, bl) - D_val(D_ols, bl)) / P),
    A_sls = summ(wA), D_sls = summ(wD), wA = wA)
  cat(sprintf("[%s] A: sep=%.3f gain=%.4f | D: sep=%.3f gain=%.4f\n", nm,
              out[[nm]]$sepA, out[[nm]]$gainA, out[[nm]]$sepD, out[[nm]]$gainD))
  print(summ(wA), row.names = FALSE)
}

# ---------- Monte Carlo check: oracle conditional SLS on the SKEW-HET field ----------
gen_skew <- function(g3) { k <- 4 / g3^2; (rgamma(length(g3), k, 1) - k) / sqrt(k) }
exact <- function(w, N) { cnt <- as.integer(round(w * N)); rep(GRID, cnt) }
fit_csls <- function(x, y, bl_x, iters = 6) {
  Fm <- fX(x)
  J <- matrix(0, P + 1, P + 1)
  J[1:P, 1:P] <- crossprod(Fm * sqrt(bl_x$a)); J[1:P, P + 1] <- J[P + 1, 1:P] <- colSums(Fm * bl_x$c)
  J[P + 1, P + 1] <- sum(bl_x$d); Ji <- solve(J)
  b <- qr.solve(Fm, y); s <- mean((y - Fm %*% b)^2)
  th <- c(b, s)
  for (i in seq_len(iters)) {           # Fisher scoring on the estimating equation
    e <- as.numeric(y - Fm %*% th[1:P]); m2 <- e^2 - th[P + 1]
    u1 <- bl_x$a * e + bl_x$c * m2; u2 <- bl_x$c * e + bl_x$d * m2
    th <- th + as.numeric(Ji %*% c(colSums(Fm * u1), sum(u2)))
  }
  th[1:P]
}
mc_design <- function(w, nm, seed) {
  fl <- fields[[nm]]; x <- exact(w, NMC); g3x <- fl$g3(x)
  bl_x <- blocks(g3x, fl$g4(x)); Fm <- fX(x)
  set.seed(seed)
  est <- t(vapply(seq_len(M), function(m) {
    y <- as.numeric(Fm %*% BETA) + gen_skew(g3x); fit_csls(x, y, bl_x)
  }, numeric(P)))
  tse <- rowSums((est - matrix(BETA, M, P, byrow = TRUE))^2)
  Jn <- Jsum(rep(1, length(x)), x, bl_x)
  list(tse = tse, mc_over_analytic = mean(tse) / sum(diag(Lsel %*% solve(Jn))))
}
cat("\n[MC] oracle conditional SLS, SKEW_HET ...\n")
mo <- mc_design(A_ols, "SKEW_HET", SEED); ms <- mc_design(out$SKEW_HET$wA, "SKEW_HET", SEED)
ok <- is.finite(mo$tse) & is.finite(ms$tse)
r  <- mean(mo$tse[ok]) / mean(ms$tse[ok])
set.seed(SEED + 1L)
bs <- replicate(3000, { i <- sample.int(sum(ok), replace = TRUE); mean(mo$tse[ok][i]) / mean(ms$tse[ok][i]) })
mc <- list(N = NMC, M = M, gain = r, lo = unname(quantile(bs, .025)), hi = unname(quantile(bs, .975)),
           ratio_ols_design = mo$mc_over_analytic, ratio_sls_design = ms$mc_over_analytic)
cat(sprintf("[MC] N=%d M=%d: MC/analytic total variance = %.3f (OLS design), %.3f (SLS design); gain = %.4f [%.4f, %.4f]\n",
            NMC, M, mc$ratio_ols_design, mc$ratio_sls_design, mc$gain, mc$lo, mc$hi))

# ---------- output ----------
res <- list(meta = list(M = M, N = NMC, seed = SEED, model = "quadratic dose-response on [0,1]", beta = BETA,
                        estimator = "conditional SLS (common sigma^2, conditional body F2(x)), oracle field"),
            A_ols = summ(A_ols), D_ols = summ(D_ols),
            fields = lapply(out, function(o) o[c("sepA", "gainA", "sepD", "gainD", "A_sls", "D_sls")]),
            mc_skew_het = mc)
write_json(res, file.path(HERE, "results/conditional_sls.json"), auto_unbox = TRUE, pretty = TRUE, digits = 6)
lab <- c(KURT_HET = "KURT (symmetric) & heteroscedastic", KURT_HOM = "KURT (symmetric) & homoscedastic",
         SKEW_HET = "SKEW (asymmetric) & heteroscedastic", SKEW_HOM = "SKEW (asymmetric) & homoscedastic")
writeLines(c("\\begin{tabular}{llcccc}", "\\toprule",
             "Regime & Cumulant field & $A$-sep & $A$-gain & $D$-sep & $D$-gain \\\\", "\\midrule",
             vapply(names(out), function(nm) sprintf("%s & %.3f & %.3f & %.3f & %.3f \\\\", lab[[nm]],
                    out[[nm]]$sepA, out[[nm]]$gainA, out[[nm]]$sepD, out[[nm]]$gainD), ""),
             "\\bottomrule", "\\end{tabular}"),
           file.path(HERE, "tables/tab_conditional_sls.tex"))
cat("saved results/conditional_sls.json, tables/tab_conditional_sls.tex\n== DONE ==\n")
