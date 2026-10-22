#!/usr/bin/env Rscript
# =====================================================================
# Definitive experiment suite for:
#   "Optimal Experimental Design for the Polynomial Maximization Method"
# Produces results/summary.json, tables/*.tex, figures/*.pdf.
#
# Engine: R + EstemPMM (CRAN). Empirical reduction-coefficient fields,
# multiplicative D-/A-optimal designs on g_S-weighted information, and a
# locally-adaptive PMM estimator validated against the global-moment one.
#
# Usage: Rscript run_all.R [M] [M_cal]      (defaults M=400, M_cal=2000)
# =====================================================================
suppressMessages({library(EstemPMM); library(jsonlite)})
HAS_GG <- requireNamespace("ggplot2", quietly = TRUE)
if (HAS_GG) suppressMessages(library(ggplot2))

args  <- commandArgs(trailingOnly = TRUE)
M     <- if (length(args) >= 1) as.integer(args[1]) else 400L
M_CAL <- if (length(args) >= 2) as.integer(args[2]) else 2000L
SEED  <- 20260606L
BETA  <- c(2.0, 1.5, -1.0)          # quadratic dose-response a, b, c
GRID  <- seq(0, 1, by = 0.02)
fX    <- function(x) cbind(1, x, x^2)
P     <- 3L
# resolve the paper root relative to this script (portable: works wherever the paper lives)
.argv <- commandArgs(FALSE); .f <- sub("^--file=", "", grep("^--file=", .argv, value = TRUE))
HERE  <- if (length(.f)) normalizePath(file.path(dirname(.f), "..")) else "."
cat(sprintf("== run_all | EstemPMM %s | M=%d M_cal=%d | seed=%d ==\n",
            as.character(packageVersion("EstemPMM")), M, M_CAL, SEED))

# ---------- error generators (centered, unit variance) ----------
# KURT regime: symmetric Beta(a,a); only kurtosis varies. a large -> ~Gaussian, a=1 -> uniform.
# A fixed-degree polynomial location estimator (PMM3) does NOT exploit the uniform hard edge,
# so it stays sqrt(n)-regular with a CONSTANT g3 down to a=1 (the irregular midrange/MLE
# super-efficiency is unavailable to a degree-3 stochastic polynomial); hence a in [1,8] is safe.
gen_kurt_param <- function(a, n) (rbeta(n, a, a) - 0.5) * 2 * sqrt(2 * a + 1)
a_of_x   <- function(x, sub) if (sub == "HET") 8 - 7 * x else rep(2.0, length(x))
# SKEW regime: centered scaled Gamma; skewness gamma3 varies.
gen_skew_param <- function(g3, n) { k <- 4 / g3^2; (rgamma(n, k, 1) - k) / sqrt(k) }
g3_of_x  <- function(x, sub) if (sub == "HET") 0.2 + 1.6 * x else rep(1.0, length(x))

# ---------- empirical reduction-coefficient calibration ----------
# g_S(param) = Var(PMM_S location)/Var(sample mean), from EstemPMM's own fits.
calib <- function(params, genloc) {
  G <- sapply(params, function(pp) {
    set.seed(SEED + as.integer(1000 * pp))
    est <- matrix(NA_real_, M_CAL, 3)
    for (m in seq_len(M_CAL)) {
      e <- genloc(pp, 400L); y <- 5 + e; df <- data.frame(y = y, x = rep(0, 400L))
      est[m, ] <- c(mean(y),
        tryCatch(coef(lm_pmm2(y ~ 1, data = df))[1], error = function(e) NA),
        tryCatch(coef(lm_pmm3(y ~ 1, data = df))[1], error = function(e) NA))
    }
    v <- apply(est, 2, var, na.rm = TRUE); c(v[2] / v[1], v[3] / v[1])
  })
  list(g2 = approxfun(params, pmin(G[1, ], 1), rule = 2),
       g3 = approxfun(params, pmin(G[2, ], 1), rule = 2),
       tab = data.frame(param = params, g2 = G[1, ], g3 = G[2, ]))
}

# ---------- optimal-design routines (transformed regressors H=f/sqrt(g)) ----------
dopt <- function(H, it = 6000, tol = 1e-10) {
  K <- nrow(H); p <- ncol(H); w <- rep(1/K, K)
  for (i in seq_len(it)) { Mi <- tryCatch(solve(crossprod(H*sqrt(w))), error=function(e) MASS::ginv(crossprod(H*sqrt(w))))
    d <- rowSums((H %*% Mi) * H); wn <- w*d/p; wn <- wn/sum(wn); if (max(abs(wn-w))<tol){w<-wn;break}; w <- wn }
  w
}
aopt <- function(H, it = 9000, tol = 1e-11) {
  K <- nrow(H); w <- rep(1/K, K)
  for (i in seq_len(it)) { Mi <- tryCatch(solve(crossprod(H*sqrt(w))), error=function(e) MASS::ginv(crossprod(H*sqrt(w))))
    dA <- rowSums((H %*% (Mi %*% Mi)) * H); wn <- w*sqrt(pmax(dA,0)); wn <- wn/sum(wn); if (max(abs(wn-w))<tol){w<-wn;break}; w <- wn }
  w
}
Minfo  <- function(w, g) { H <- fX(GRID)/sqrt(g); crossprod(H*sqrt(w)) }
logdet <- function(Mat) sum(log(pmax(eigen(Mat, symmetric=TRUE, only.values=TRUE)$values, .Machine$double.eps)))
atrace <- function(w, g) sum(diag(solve(Minfo(w, g))))
summ   <- function(w, thr = 1e-3) { k <- which(w > thr); data.frame(x = GRID[k], w = round(w[k], 3)) }
exact  <- function(w, N) { cnt <- as.integer(round(w*N)); cnt[cnt<0] <- 0; rep(GRID, cnt) }

# ---------- estimators (global vs locally-adaptive PMM_S) ----------
# `seed` sets common random numbers so OLS-design and PMM-design comparisons are
# paired (identical error streams) -> HOM controls return exactly 1.000.
# return per-replication total squared error ||theta_hat - theta||^2 (paired across designs
# via the shared seed) so the design gain ratio gets a paired bootstrap CI.
fit_global <- function(gen, doses, M, deg, seed) {
  set.seed(seed); tse <- rep(NA_real_, M)
  for (m in seq_len(M)) {
    x <- doses; y <- BETA[1] + BETA[2]*x + BETA[3]*x^2 + gen(x)
    df <- data.frame(y = y, x = x)
    th <- if (deg == 3) tryCatch(coef(lm_pmm3(y~x+I(x^2), data=df)), error=function(e) rep(NA,P))
          else          tryCatch(coef(lm_pmm2(y~x+I(x^2), data=df)), error=function(e) rep(NA,P))
    tse[m] <- sum((th - BETA)^2)
  }
  list(tse = tse, totvar = mean(tse, na.rm=TRUE), conv = mean(!is.na(tse)))
}
# locally-adaptive PMM_S: fit PMM_S location per dose group, then combine the per-dose
# estimates by INVERSE-VARIANCE weights w_k = n_k / g_S(x_k). Since Var(m_hat_k) ~
# g_S(x_k) sigma^2 / n_k, this gives Cov(theta_hat) = sigma^2 M_{g_S}(design)^{-1} exactly,
# so the g_S-weighted optimal design genuinely minimizes its variance (design/estimator matched).
# g3fun(x) supplies the per-dose reduction (here the calibrated field; in practice estimable
# per group from the same PMM_S fit -- the simulation isolates the design effect from weight noise).
fit_local <- function(gen, doses, M, deg, seed, g3fun) {
  set.seed(seed)
  grp <- sort(unique(doses)); X <- fX(grp); nk <- as.integer(table(factor(doses, levels=as.character(grp))))
  wk <- nk / pmin(g3fun(grp), 1); W <- diag(wk); tse <- rep(NA_real_, M)
  loc <- function(yg) if (length(yg) < 25) mean(yg) else {
    df <- data.frame(yg=yg)
    if (deg==3) tryCatch(coef(lm_pmm3(yg~1,data=df))[1], error=function(e) mean(yg))
    else        tryCatch(coef(lm_pmm2(yg~1,data=df))[1], error=function(e) mean(yg)) }
  for (m in seq_len(M)) {
    x <- doses; y <- BETA[1] + BETA[2]*x + BETA[3]*x^2 + gen(x)
    mk <- sapply(grp, function(g) loc(y[x == g]))
    th <- tryCatch(as.numeric(solve(t(X)%*%W%*%X, t(X)%*%W%*%mk)), error=function(e) rep(NA,P))
    tse[m] <- sum((th - BETA)^2)
  }
  list(tse = tse, totvar = mean(tse, na.rm=TRUE), conv = mean(!is.na(tse)))
}
# paired bootstrap CI for the gain ratio mean(tse_ols)/mean(tse_pmm)
gain_ci <- function(tse_ols, tse_pmm, B = 3000) {
  ok <- is.finite(tse_ols) & is.finite(tse_pmm); a <- tse_ols[ok]; b <- tse_pmm[ok]; n <- length(a)
  r <- mean(a) / mean(b)
  bs <- replicate(B, { idx <- sample.int(n, n, replace = TRUE); mean(a[idx]) / mean(b[idx]) })
  c(gain = r, lo = unname(quantile(bs, .025)), hi = unname(quantile(bs, .975)))
}

# ---------- design-builder for a regime ----------
designs_for <- function(gfield, sub) {
  g2 <- pmin(gfield$g2_x(GRID, sub), 1); g3 <- pmin(gfield$g3_x(GRID, sub), 1)
  list(g2=g2, g3=g3,
       A_ols=aopt(fX(GRID)), A_p2=aopt(fX(GRID)/sqrt(g2)), A_p3=aopt(fX(GRID)/sqrt(g3)),
       D_ols=dopt(fX(GRID)), D_p2=dopt(fX(GRID)/sqrt(g2)), D_p3=dopt(fX(GRID)/sqrt(g3)))
}
sep <- function(w1, w2) 0.5 * sum(abs(w1 - w2))

# =====================================================================
# EXPERIMENT A — KURT regime (the headline: S>=3 reserve, 2nd-order blind)
# =====================================================================
cat("\n[E-KURT] calibrating g2,g3 over Beta shape a ...\n")
cal_k <- calib(c(8,5,3.5,2.5,2,1.6,1.3,1.0), gen_kurt_param)
print(round(cal_k$tab, 3), row.names = FALSE)
gfield_k <- list(g2_x = function(x,sub) cal_k$g2(a_of_x(x,sub)),
                 g3_x = function(x,sub) cal_k$g3(a_of_x(x,sub)))
DK_het <- designs_for(gfield_k, "HET"); DK_hom <- designs_for(gfield_k, "HOM")

kurt_analytic <- function(D) list(
  # A-criterion (headline): separation + efficiency on the A-optimal designs
  sep_p2 = sep(D$A_p2, D$A_ols), sep_p3 = sep(D$A_p3, D$A_ols),
  gainA_p3 = atrace(D$A_ols, D$g3) / atrace(D$A_p3, D$g3),
  # D-criterion (added): separation + efficiency on the D-optimal designs
  sepD_p2 = sep(D$D_p2, D$D_ols), sepD_p3 = sep(D$D_p3, D$D_ols),
  gainD_p3 = exp((logdet(Minfo(D$D_p3, D$g3)) - logdet(Minfo(D$D_ols, D$g3))) / P))
AK_het <- kurt_analytic(DK_het); AK_hom <- kurt_analytic(DK_hom)
cat(sprintf("[E-KURT] HET A-crit: PMM2-sep=%.3f PMM3-sep=%.3f gain=%.4f | HET D-crit: PMM2-sep=%.3f PMM3-sep=%.3f gain=%.4f | HOM A-sep=%.3f D-sep=%.3f\n",
            AK_het$sep_p2, AK_het$sep_p3, AK_het$gainA_p3, AK_het$sepD_p2, AK_het$sepD_p3, AK_het$gainD_p3, AK_hom$sep_p3, AK_hom$sepD_p3))

# n-sweep: realized total-MSE gain of PMM3-design vs OLS-design, global & local, with CI
NS <- c(300,600,900,1200)
gen_k_het <- function(x) gen_kurt_param(a_of_x(x,"HET"), length(x))
gen_k_hom <- function(x) gen_kurt_param(a_of_x(x,"HOM"), length(x))
g3fun_het <- function(x) pmin(cal_k$g3(a_of_x(x,"HET")), 1)
g3fun_hom <- function(x) pmin(cal_k$g3(a_of_x(x,"HOM")), 1)
nsweep <- lapply(NS, function(N){
  sg <- SEED + N; sl <- SEED + N + 1L         # paired CRN within each estimator
  go <- fit_global(gen_k_het, exact(DK_het$A_ols,N), M, 3, sg); gp <- fit_global(gen_k_het, exact(DK_het$A_p3,N), M, 3, sg)
  lo <- fit_local (gen_k_het, exact(DK_het$A_ols,N), M, 3, sl, g3fun_het); lp <- fit_local (gen_k_het, exact(DK_het$A_p3,N), M, 3, sl, g3fun_het)
  cg <- gain_ci(go$tse, gp$tse); cl <- gain_ci(lo$tse, lp$tse)
  data.frame(N=N, global=cg["gain"], local=cl["gain"], local_lo=cl["lo"], local_hi=cl["hi"],
             conv=min(go$conv,gp$conv,lo$conv,lp$conv))
})
nsweep <- do.call(rbind, nsweep); rownames(nsweep) <- NULL
cat("[E-KURT] n-sweep (PMM3-design vs OLS-design total-MSE gain; local 95% CI):\n"); print(round(nsweep,3), row.names=FALSE)
# HOM degeneracy control at one N (paired CRN -> exactly 1.000 if designs coincide)
ho <- fit_local(gen_k_hom, exact(DK_hom$A_ols,600), M, 3, SEED+777L, g3fun_hom); hp <- fit_local(gen_k_hom, exact(DK_hom$A_p3,600), M, 3, SEED+777L, g3fun_hom)
kurt_hom_localgain <- ho$totvar/hp$totvar
cat(sprintf("[E-KURT] HOM degeneracy control (N=600, local): gain=%.4f (expect 1.000)\n", kurt_hom_localgain))

# =====================================================================
# EXPERIMENT B — SKEW regime (mechanism + flat-D caveat, 2nd-order visible)
# =====================================================================
cat("\n[E-SKEW] calibrating g2,g3 over Gamma skewness ...\n")
cal_s <- calib(c(0.2,0.5,0.8,1.1,1.4,1.6,1.8), gen_skew_param)
print(round(cal_s$tab,3), row.names=FALSE)
gfield_s <- list(g2_x=function(x,sub) cal_s$g2(g3_of_x(x,sub)), g3_x=function(x,sub) cal_s$g3(g3_of_x(x,sub)))
DS_het <- designs_for(gfield_s,"HET"); DS_hom <- designs_for(gfield_s,"HOM")
skew_analytic <- function(D) list(
  sep_p2=sep(D$A_p2,D$A_ols),
  gainA_p2=atrace(D$A_ols,D$g2)/atrace(D$A_p2,D$g2),
  gainD_p2=exp((logdet(Minfo(D$A_p2,D$g2))-logdet(Minfo(D$A_ols,D$g2)))/P))
AS_het <- skew_analytic(DS_het); AS_hom <- skew_analytic(DS_hom)
gen_s_het <- function(x) gen_skew_param(g3_of_x(x,"HET"), length(x))
g2fun_s_het <- function(x) pmin(cal_s$g2(g3_of_x(x,"HET")), 1)   # ~= 1 (skewness location-blind)
so_g <- fit_global(gen_s_het, exact(DS_het$A_ols,600),M,2,SEED+222L); sp_g <- fit_global(gen_s_het, exact(DS_het$A_p2,600),M,2,SEED+222L)
so_l <- fit_local (gen_s_het, exact(DS_het$A_ols,600),M,2,SEED+223L,g2fun_s_het); sp_l <- fit_local (gen_s_het, exact(DS_het$A_p2,600),M,2,SEED+223L,g2fun_s_het)
skew_mc <- list(global=so_g$totvar/sp_g$totvar, local=so_l$totvar/sp_l$totvar)
cat(sprintf("[E-SKEW] HET: PMM2-sep=%.3f analyticA-gain=%.4f | MC gain global=%.3f local=%.3f | HOM-sep=%.3f\n",
            AS_het$sep_p2, AS_het$gainA_p2, skew_mc$global, skew_mc$local, AS_hom$sep_p2))

# =====================================================================
# EXPERIMENT C — separation vs heteroscedasticity strength (KURT)
# =====================================================================
cat("\n[E-STRENGTH] design separation vs kurtosis-heteroscedasticity slope ...\n")
slopes <- seq(0, 7, by = 0.5)   # a(x)=8 - slope*x ; slope=0 -> homoscedastic
strength <- lapply(slopes, function(sl){
  ax <- pmax(8 - sl*GRID, 1.0)
  g2 <- pmin(cal_k$g2(ax),1); g3 <- pmin(cal_k$g3(ax),1)
  data.frame(slope=sl,
             het = max(g3)-min(g3),
             sep_p2 = sep(aopt(fX(GRID)/sqrt(g2)), aopt(fX(GRID))),
             sep_p3 = sep(aopt(fX(GRID)/sqrt(g3)), aopt(fX(GRID))))
})
strength <- do.call(rbind, strength)
print(round(strength,3), row.names=FALSE)

# =====================================================================
# OUTPUT: JSON, tables, figures
# =====================================================================
res <- list(meta=list(engine=as.character(packageVersion("EstemPMM")), M=M, M_cal=M_CAL, seed=SEED,
                       model="quadratic dose-response on [0,1]", beta=BETA),
  kurt=list(calib=cal_k$tab,
            het=list(sep_p2=AK_het$sep_p2, sep_p3=AK_het$sep_p3, gainA=AK_het$gainA_p3, gainD=AK_het$gainD_p3,
                     sepD_p2=AK_het$sepD_p2, sepD_p3=AK_het$sepD_p3,
                     A_ols=summ(DK_het$A_ols), A_p3=summ(DK_het$A_p3),
                     D_ols=summ(DK_het$D_ols), D_p3=summ(DK_het$D_p3)),
            hom=list(sep_p3=AK_hom$sep_p3, sepD_p3=AK_hom$sepD_p3, local_gain=kurt_hom_localgain),
            nsweep=nsweep),
  skew=list(calib=cal_s$tab,
            het=list(sep_p2=AS_het$sep_p2, gainA=AS_het$gainA_p2, gainD=AS_het$gainD_p2,
                     mc_global=skew_mc$global, mc_local=skew_mc$local),
            hom=list(sep_p2=AS_hom$sep_p2)),
  strength=strength)
write_json(res, file.path(HERE,"results/summary.json"), auto_unbox=TRUE, pretty=TRUE, digits=6)
cat("\nsaved results/summary.json\n")

## ---- LaTeX tables (hand-built; no extra deps) ----
wt <- function(path, lines) writeLines(lines, file.path(HERE, path))
# Table: KURT calibration (g2==1, g3 dropping)
ck <- cal_k$tab
wt("tables/tab_calib_kurt.tex", c(
  "\\begin{tabular}{rrr}", "\\toprule",
  "Beta shape $a$ & $g_2(a)$ & $g_3(a)$ \\\\", "\\midrule",
  paste0(sprintf("%.1f", ck$param), " & ", sprintf("%.3f", ck$g2), " & ", sprintf("%.3f", ck$g3), " \\\\"),
  "\\bottomrule", "\\end{tabular}"))
# Table: separation summary
wt("tables/tab_separation.tex", c(
  "\\begin{tabular}{llccc}", "\\toprule",
  "Regime & Cumulant field & PMM$_2$-sep & PMM$_3$-sep & analytic $A$-gain \\\\", "\\midrule",
  sprintf("KURT (symmetric) & heteroscedastic & %.3f & %.3f & %.3f \\\\", AK_het$sep_p2, AK_het$sep_p3, AK_het$gainA_p3),
  sprintf("KURT (symmetric) & homoscedastic & %.3f & %.3f & %.3f \\\\", AK_hom$sep_p2, AK_hom$sep_p3, AK_hom$gainA_p3),
  sprintf("SKEW (asymmetric) & heteroscedastic & %.3f & --- & %.3f \\\\", AS_het$sep_p2, AS_het$gainA_p2),
  sprintf("SKEW (asymmetric) & homoscedastic & %.3f & --- & %.3f \\\\", AS_hom$sep_p2, 1.0),
  "\\bottomrule", "\\end{tabular}"))
# Table: criterion comparison (A vs D) for the KURT-heteroscedastic regime
wt("tables/tab_criterion.tex", c(
  "\\begin{tabular}{lccc}", "\\toprule",
  "Criterion & PMM$_2$ separation & PMM$_3$ separation & analytic PMM$_3$ gain \\\\", "\\midrule",
  sprintf("$A$ (total coef.\\ MSE) & %.3f & %.3f & %.3f \\\\", AK_het$sep_p2, AK_het$sep_p3, AK_het$gainA_p3),
  sprintf("$D$ (gen.\\ variance) & %.3f & %.3f & %.3f \\\\", AK_het$sepD_p2, AK_het$sepD_p3, AK_het$gainD_p3),
  "\\bottomrule", "\\end{tabular}"))
# Table: n-sweep
wt("tables/tab_nsweep.tex", c(
  "\\begin{tabular}{rcc}", "\\toprule",
  "Total $N$ & global PMM$_3$ & locally-adaptive PMM$_3$ (95\\% CI) \\\\", "\\midrule",
  paste0(nsweep$N, " & ", sprintf("%.3f", nsweep$global), " & ",
         sprintf("%.3f", nsweep$local), " [", sprintf("%.3f", nsweep$local_lo), ", ",
         sprintf("%.3f", nsweep$local_hi), "] \\\\"),
  "\\bottomrule", "\\end{tabular}"))
cat("saved tables/*.tex\n")

## ---- figures ----
if (HAS_GG) tryCatch({
  th <- theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank())
  # Fig 1: reduction fields + designs (KURT-HET)
  gx <- data.frame(x=GRID, g2=DK_het$g2, g3=DK_het$g3)
  f1a <- ggplot(gx, aes(x)) +
    geom_line(aes(y=g2, colour="g[2](x)"), linewidth=1) +
    geom_line(aes(y=g3, colour="g[3](x)"), linewidth=1) +
    scale_colour_manual(values=c("g[2](x)"="grey50","g[3](x)"="black"),
                        labels=c(expression(g[2](x)), expression(g[3](x))), name=NULL) +
    labs(x="dose x", y="reduction coefficient") + ylim(0,1.02) + th + theme(legend.position="top")
  dd <- rbind(transform(summ(DK_het$A_ols), design="OLS / PMM2-optimal"),
              transform(summ(DK_het$A_p3), design="PMM3-optimal"))
  f1b <- ggplot(dd, aes(x, w, colour=design)) +
    geom_segment(aes(xend=x, yend=0), position=position_dodge(width=.02), linewidth=1) +
    geom_point(size=2.4, position=position_dodge(width=.02)) +
    scale_colour_manual(values=c("OLS / PMM2-optimal"="grey55","PMM3-optimal"="black"), name=NULL) +
    labs(x="dose x", y="design weight") + th + theme(legend.position="top")
  ggsave(file.path(HERE,"figures/fig_reserve_fields.pdf"), f1a, width=5, height=3.4)
  ggsave(file.path(HERE,"figures/fig_reserve_designs.pdf"), f1b, width=5, height=3.4)
  # Fig 2: efficiency vs n (local with bootstrap CI ribbon)
  nl <- rbind(data.frame(N=nsweep$N, gain=nsweep$global, lo=NA, hi=NA, estimator="global PMM3"),
              data.frame(N=nsweep$N, gain=nsweep$local, lo=nsweep$local_lo, hi=nsweep$local_hi,
                         estimator="locally-adaptive PMM3"))
  f2 <- ggplot(nl, aes(N, gain, colour=estimator, shape=estimator)) +
    geom_hline(yintercept=1, linetype=2, colour="grey60") +
    geom_ribbon(aes(ymin=lo, ymax=hi, fill=estimator), alpha=.15, colour=NA) +
    geom_line(linewidth=.8) + geom_point(size=2.6) +
    scale_colour_manual(values=c("global PMM3"="grey55","locally-adaptive PMM3"="black"), name=NULL) +
    scale_fill_manual(values=c("global PMM3"="grey55","locally-adaptive PMM3"="black"), name=NULL, guide="none") +
    scale_shape_manual(values=c(1,16), name=NULL) +
    labs(x="total sample size N", y="design efficiency gain\n(PMM3-optimal vs OLS-optimal)") +
    th + theme(legend.position="top")
  ggsave(file.path(HERE,"figures/fig_efficiency_vs_n.pdf"), f2, width=5.4, height=3.6)
  # Fig 3: separation vs heteroscedasticity strength
  sl <- rbind(data.frame(het=strength$het, sep=strength$sep_p2, order="PMM2 (2nd order)"),
              data.frame(het=strength$het, sep=strength$sep_p3, order="PMM3 (3rd order)"))
  f3 <- ggplot(sl, aes(het, sep, colour=order, shape=order)) +
    geom_line(linewidth=.8) + geom_point(size=2.6) +
    scale_colour_manual(values=c("PMM2 (2nd order)"="grey55","PMM3 (3rd order)"="black"), name=NULL) +
    scale_shape_manual(values=c(1,16), name=NULL) +
    labs(x=expression("kurtosis-heteroscedasticity strength  "*(max~g[3]-min~g[3])),
         y="design separation (A, L1/2)") + th + theme(legend.position="top")
  ggsave(file.path(HERE,"figures/fig_separation_vs_strength.pdf"), f3, width=5.4, height=3.6)
  cat("saved figures/*.pdf\n")
}, error = function(e) cat("FIG WARN:", conditionMessage(e), "\n"))

cat("\n== DONE ==\n")
