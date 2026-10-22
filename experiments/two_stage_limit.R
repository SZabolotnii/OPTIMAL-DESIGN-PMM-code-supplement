#!/usr/bin/env Rscript
# Large-N limit of the two-stage design of experiments/two_stage.R.
# If the pilot field estimate is exact (its N -> infinity limit), stage 2 fills the positive deficits
# of N * w_PMM3 left by a pilot of fraction r on the OLS A-optimal design. The pilot mass on the OLS
# interior dose 0.50 stays in the final design, so for fixed r > 0 the combined design does NOT reach
# the oracle PMM3 design. This prints its A-gain under the calibrated field for several r.
# Usage (from the repository root): Rscript experiments/two_stage_limit.R > experiments/two_stage_limit.log
suppressMessages(library(jsonlite))
GRID <- seq(0, 1, by = 0.02); fX <- function(x) cbind(1, x, x^2)
aopt <- function(H, it = 9000, tol = 1e-11) { K <- nrow(H); w <- rep(1/K, K)
  for (i in seq_len(it)) { Mi <- solve(crossprod(H*sqrt(w))); dA <- rowSums((H %*% (Mi %*% Mi)) * H)
    wn <- w*sqrt(pmax(dA,0)); wn <- wn/sum(wn); if (max(abs(wn-w))<tol){w<-wn;break}; w <- wn }; w }
atrace <- function(w, g) { H <- fX(GRID)/sqrt(g); sum(diag(solve(crossprod(H*sqrt(w))))) }
cal <- fromJSON("results/summary.json")$kurt$calib
g3c <- approxfun(cal$param, pmin(cal$g3, 1), rule = 2)
g <- pmin(g3c(8 - 7*GRID), 1); Ao <- aopt(fX(GRID)); Ap <- aopt(fX(GRID)/sqrt(g))
cat(sprintf("oracle one-stage A-gain %.4f\n", atrace(Ao, g)/atrace(Ap, g)))
for (r in c(1/3, 0.2, 0.1, 0.05)) {
  c1 <- r*Ao; def <- pmax(Ap - c1, 0); w <- c1 + def/sum(def)*(1 - r)
  cat(sprintf("pilot fraction %.3f: limit two-stage design A-gain %.4f (share %.0f%%)\n",
              r, atrace(Ao, g)/atrace(w, g), 100*(atrace(Ao, g)/atrace(w, g) - 1)/(atrace(Ao, g)/atrace(Ap, g) - 1)))
}
