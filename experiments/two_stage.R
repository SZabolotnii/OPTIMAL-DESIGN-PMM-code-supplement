#!/usr/bin/env Rscript
# =====================================================================
# Two-stage (feasible) PMM_3 design in the KURT-heteroscedastic regime.
#
# The paper's +4.7% is an oracle-field gain: the design and the GLS weights use the
# calibrated g_3 field. Here the field is ESTIMATED from a pilot:
#   stage 1  N1 = round(r*N) observations on the OLS A-optimal design (known in advance);
#            per pilot stratum, plug-in g3_hat = 1 - g4_hat^2/(6+9 g4_hat+g6_hat)
#            (EstemPMM's own formula), clipped to [0.05, 1]; field over the grid by
#            linear interpolation in x between the pilot doses (constant beyond).
#   design   A-optimal design for the estimated field (multiplicative algorithm on h = f/sqrt(g3_hat)).
#   stage 2  N - N1 observations added so that the combined allocation approaches N * w_hat
#            (augmentation: deficits filled, largest remainder).
#   estimate locally-adaptive PMM_3 on ALL N observations (lm_pmm3 per dose, mean if n_k < 25),
#            GLS weights n_k / g3_hat_k with g3_hat_k from the final stratum fit (feasible).
# The same two-stage data are also pooled with ORACLE weights (calibrated field), which separates the
# loss from estimating the design from the loss from estimating the pooling weights.
# Comparators with the same N and the same estimator: one-stage OLS-optimal design (O) and
# the oracle one-stage PMM_3-optimal design of the paper (P). Paired: each replication uses the
# same seed in every arm. Output: per-replication CSV + summary CSV; nothing is tuned to the result.
#
# Usage: Rscript experiments/two_stage.R [M] [cores]     (default M = 2500, cores = 8)
# =====================================================================
suppressMessages({library(EstemPMM); library(jsonlite); library(parallel)})
args  <- commandArgs(trailingOnly = TRUE)
M     <- if (length(args) >= 1) as.integer(args[1]) else 2500L
CORES <- if (length(args) >= 2) as.integer(args[2]) else 8L
SEED  <- 20261021L
NS    <- c(600L, 900L, 1200L, 2400L)
R1    <- 1/3                               # pilot fraction
BETA  <- c(2.0, 1.5, -1.0); GRID <- seq(0, 1, by = 0.02); P <- 3L
fX    <- function(x) cbind(1, x, x^2)
.argv <- commandArgs(FALSE); .f <- sub("^--file=", "", grep("^--file=", .argv, value = TRUE))
HERE  <- if (length(.f)) normalizePath(file.path(dirname(.f), "..")) else "."
RNGkind("L'Ecuyer-CMRG")
cat(sprintf("== two_stage | EstemPMM %s | M=%d | seed=%d | pilot fraction %.3f ==\n",
            as.character(packageVersion("EstemPMM")), M, SEED, R1))

gen_kurt_param <- function(a, n) (rbeta(n, a, a) - 0.5) * 2 * sqrt(2 * a + 1)
a_of_x <- function(x) 8 - 7 * x
gen    <- function(x) gen_kurt_param(a_of_x(x), length(x))

# same multiplicative A-algorithm as run_all.R (optional warm start)
aopt <- function(H, it = 9000, tol = 1e-11, w0 = NULL) {
  K <- nrow(H); w <- if (is.null(w0)) rep(1/K, K) else pmax(w0, 1e-6) / sum(pmax(w0, 1e-6))
  for (i in seq_len(it)) { Mi <- tryCatch(solve(crossprod(H*sqrt(w))), error=function(e) MASS::ginv(crossprod(H*sqrt(w))))
    dA <- rowSums((H %*% (Mi %*% Mi)) * H); wn <- w*sqrt(pmax(dA,0)); wn <- wn/sum(wn); if (max(abs(wn-w))<tol){w<-wn;break}; w <- wn }
  w
}
atrace <- function(w, g) { H <- fX(GRID)/sqrt(g); sum(diag(solve(crossprod(H*sqrt(w))))) }
round_alloc <- function(target, n) {        # nonnegative integer vector summing to n
  target <- pmax(target, 0); if (sum(target) == 0) target <- rep(1, length(target))
  t <- target / sum(target) * n; c0 <- floor(t); r <- n - sum(c0)
  if (r > 0) { o <- order(t - c0, decreasing = TRUE)[seq_len(r)]; c0[o] <- c0[o] + 1 }
  c0
}
doses_of <- function(cnt) rep(GRID, cnt)

# oracle (calibrated, n = 400) field from the stored calibration, exactly as run_all.R
cal  <- fromJSON(file.path(HERE, "results/summary.json"))$kurt$calib
g3c  <- approxfun(cal$param, pmin(cal$g3, 1), rule = 2)
gcal <- pmin(g3c(a_of_x(GRID)), 1)
A_ols <- aopt(fX(GRID)); A_p3 <- aopt(fX(GRID)/sqrt(gcal))
cat(sprintf("oracle analytic A-gain (calibrated field) = %.4f\n", atrace(A_ols, gcal)/atrace(A_p3, gcal)))
# asymptotic closed-form field (Appendix B): g3 = 1 - 6(u+2)/(u(u^2-7u+22)), u = 2a+3
g3_closed <- function(a) { u <- 2*a + 3; 1 - 6*(u+2)/(u*(u^2 - 7*u + 22)) }
gcf  <- g3_closed(a_of_x(GRID)); A_cf <- aopt(fX(GRID)/sqrt(gcf))
gain_cf <- atrace(A_ols, gcf)/atrace(A_cf, gcf)
sup <- function(w) { k <- which(w > 1e-3); paste(sprintf("%.2f:%.3f", GRID[k], w[k]), collapse = " ") }
cat(sprintf("closed-form field: A-opt design %s | analytic A-gain = %.4f\n", sup(A_cf), gain_cf))

g3_plugin <- function(e) {                 # EstemPMM's moment formula on centred data
  r <- e - mean(e); m2 <- mean(r^2); m4 <- mean(r^4); m6 <- mean(r^6)
  g4 <- m4/m2^2 - 3; g6 <- m6/m2^3 - 15*m4/m2^2 + 30; d <- 6 + 9*g4 + g6
  g <- if (!is.finite(d) || d <= 0) 1 else 1 - g4^2/d
  min(max(g, 0.05), 1)
}
loc_fit <- function(yg) if (length(yg) < 25) list(mu = mean(yg), g = 1) else {
  f <- tryCatch(suppressWarnings(lm_pmm3(yg ~ 1, data = data.frame(yg = yg))), error = function(e) NULL)
  if (is.null(f)) list(mu = mean(yg), g = 1) else list(mu = unname(coef(f)[1]), g = min(max(f@g_coefficient, 0.05), 1))
}
# locally-adaptive PMM_3 with GLS weights n_k/g_k; returns the total squared error under
# oracle weights (calibrated field) and under plug-in weights (g from each stratum fit)
fit_local <- function(x, y) {
  grp <- sort(unique(x)); nk <- as.integer(table(factor(x, levels = grp)))
  fits <- lapply(grp, function(g) loc_fit(y[x == g]))
  mk <- vapply(fits, `[[`, 0, "mu"); X <- fX(grp)
  tse <- function(gk) { W <- diag(nk/gk, length(grp))
    th <- tryCatch(as.numeric(solve(t(X) %*% W %*% X, t(X) %*% W %*% mk)), error = function(e) rep(NA_real_, P))
    sum((th - BETA)^2) }
  c(oracle = tse(pmin(g3c(a_of_x(grp)), 1)), plugin = tse(vapply(fits, `[[`, 0, "g")))
}

one_rep <- function(m, N) {
  N1 <- round(R1 * N); c1 <- round_alloc(A_ols, N1)
  # ---- arm O: one-stage OLS-optimal; arm P: oracle one-stage PMM3-optimal
  set.seed(SEED + 100000L * N + m); xO <- doses_of(round_alloc(A_ols, N)); yO <- fX(xO) %*% BETA + gen(xO)
  set.seed(SEED + 100000L * N + m); xP <- doses_of(round_alloc(A_p3,  N)); yP <- fX(xP) %*% BETA + gen(xP)
  tO <- fit_local(xO, yO)[["oracle"]]; tP <- fit_local(xP, yP)[["oracle"]]   # 3-point designs: weights irrelevant
  # ---- arm T: two-stage feasible
  set.seed(SEED + 100000L * N + m)
  x1 <- doses_of(c1); y1 <- fX(x1) %*% BETA + gen(x1)
  pd <- sort(unique(x1)); gp <- vapply(pd, function(d) g3_plugin(y1[x1 == d]), 0)
  ghat <- approx(pd, gp, xout = GRID, rule = 2)$y
  w_hat <- aopt(fX(GRID)/sqrt(ghat), w0 = A_ols)
  c2 <- round_alloc(pmax(w_hat * N - c1, 0), N - N1)
  x2 <- doses_of(c2); y2 <- fX(x2) %*% BETA + gen(x2)
  x <- c(x1, x2); y <- c(y1, y2)
  fT  <- fit_local(x, y); tT <- fT[["plugin"]]; tTo <- fT[["oracle"]]
  wT  <- (c1 + c2) / N                     # realised combined allocation
  c(m = m, N = N, tse_O = tO, tse_P = tP, tse_T = tT, tse_T_oraclew = tTo,
    effA_T = atrace(A_ols, gcal) / atrace(wT, gcal),          # asymptotic A-gain of the realised design
    sep_hat = 0.5 * sum(abs(w_hat - A_ols)), sep_hat_vs_oracle = 0.5 * sum(abs(w_hat - A_p3)),
    n_strata = length(unique(x)))
}

gain_ci <- function(a, b, B = 3000) {
  ok <- is.finite(a) & is.finite(b); a <- a[ok]; b <- b[ok]; n <- length(a)
  set.seed(SEED + 7L); bs <- replicate(B, { i <- sample.int(n, n, TRUE); mean(a[i]) / mean(b[i]) })
  c(gain = mean(a)/mean(b), lo = unname(quantile(bs, .025)), hi = unname(quantile(bs, .975)))
}

t0 <- Sys.time(); all <- list()
for (N in NS) {
  rows <- mclapply(seq_len(M), one_rep, N = N, mc.cores = CORES, mc.set.seed = FALSE)
  bad <- vapply(rows, function(r) inherits(r, "try-error") || is.null(r), TRUE)
  if (any(bad)) stop(sprintf("N=%d: %d replications failed", N, sum(bad)))
  all[[as.character(N)]] <- as.data.frame(do.call(rbind, rows))
  d <- all[[as.character(N)]]
  # NB: the 2026-10-21 log was written before this format gained its fourth ratio, so its
  # "effA_T mean" field there is T(oracle w)/O; results/two_stage_summary.csv is authoritative.
  cat(sprintf("N=%4d done (%.1f min): gain P/O %.3f, T/O %.3f, T(oracle w)/O %.3f, effA_T mean %.4f\n", N,
              as.numeric(difftime(Sys.time(), t0, units = "mins")),
              mean(d$tse_O)/mean(d$tse_P), mean(d$tse_O)/mean(d$tse_T), mean(d$tse_O)/mean(d$tse_T_oraclew), mean(d$effA_T)))
}
reps <- do.call(rbind, all)
write.csv(reps, file.path(HERE, "results/two_stage_reps.csv"), row.names = FALSE)
summ <- do.call(rbind, lapply(all, function(d) {
  gP <- gain_ci(d$tse_O, d$tse_P); gT <- gain_ci(d$tse_O, d$tse_T); gTP <- gain_ci(d$tse_P, d$tse_T)
  gTo <- gain_ci(d$tse_O, d$tse_T_oraclew)
  data.frame(N = d$N[1], M = nrow(d),
             oracle_gain = gP["gain"], oracle_lo = gP["lo"], oracle_hi = gP["hi"],
             twostage_gain = gT["gain"], twostage_lo = gT["lo"], twostage_hi = gT["hi"],
             twostage_oraclew_gain = gTo["gain"], tsow_lo = gTo["lo"], tsow_hi = gTo["hi"],
             twostage_vs_oracle = gTP["gain"], tvo_lo = gTP["lo"], tvo_hi = gTP["hi"],
             effA_T_mean = mean(d$effA_T), effA_T_q10 = unname(quantile(d$effA_T, .1)),
             effA_T_q90 = unname(quantile(d$effA_T, .9)),
             sep_hat_mean = mean(d$sep_hat), mean_strata = mean(d$n_strata))
}))
rownames(summ) <- NULL
write.csv(summ, file.path(HERE, "results/two_stage_summary.csv"), row.names = FALSE)
write_json(list(seed = SEED, M = M, pilot_fraction = R1, NS = NS,
                oracle_analytic_gain_calibrated = atrace(A_ols, gcal)/atrace(A_p3, gcal),
                closed_form_field = list(design = sup(A_cf), analytic_gain = gain_cf),
                engine = as.character(packageVersion("EstemPMM"))),
           file.path(HERE, "results/two_stage_meta.json"), auto_unbox = TRUE, pretty = TRUE, digits = 6)
print(round(summ, 4), row.names = FALSE)
wt <- function(path, lines) writeLines(lines, file.path(HERE, path))
wt("tables/tab_twostage.tex", c(
  "\\begin{tabular}{rcccc}", "\\toprule",
  " & oracle & \\multicolumn{2}{c}{two-stage design} & design-only \\\\",
  "Total $N$ & one-stage & plug-in weights & oracle weights & gain \\\\", "\\midrule",
  paste0(summ$N, " & ", sprintf("%.3f [%.3f, %.3f]", summ$oracle_gain, summ$oracle_lo, summ$oracle_hi), " & ",
         sprintf("%.3f [%.3f, %.3f]", summ$twostage_gain, summ$twostage_lo, summ$twostage_hi), " & ",
         sprintf("%.3f [%.3f, %.3f]", summ$twostage_oraclew_gain, summ$tsow_lo, summ$tsow_hi), " & ",
         sprintf("%.3f", summ$effA_T_mean), " \\\\"),
  "\\bottomrule", "\\end{tabular}"))
cat("saved results/two_stage_{reps,summary}.csv, results/two_stage_meta.json, tables/tab_twostage.tex\n== DONE ==\n")
