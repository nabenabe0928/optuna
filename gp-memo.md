**Short answer.** In principle every acquisition function in `acqf.py` can be computed without QMC, because they all integrate piecewise-linear or indicator utilities against Gaussians. In practice they split into three groups:

1. **Univariate Φ/φ is enough:** `LogEI`, `LogPI`, `UCB`, `LCB` and `LogCEI` are already analytic. `LogEHVI` and `LogCEHVI` can be made analytic with only univariate Φ/φ.
2. **Needs multivariate normal CDFs:** `qLogEI`, `qLogPI`, `qLogCEI`, `qLogEHVI` and `qLogCEHVI`. Here q = n_running + 1, because the running trials' fantasies sit inside the same max or HVI as the candidate.
3. **The catch:** an MVN CDF in dimension ≥ 4 has no closed form. The standard Genz–Bretz algorithm (what `scipy.stats.multivariate_normal.cdf` uses, and what the paper used) is itself randomized QMC. So "QMC-free" really holds only for small q.

## Per acquisition function

| acqf                           | QMC now?             | Analytic form                                                                                                                                                                                                                                                                    | What it needs                                                                                                                                                                                                         |
| ------------------------------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LogEI`, `LogPI`, `UCB`, `LCB` | no                   | already analytic                                                                                                                                                                                                                                                                 | `erfc`, `erfcx`, `log_ndtr`                                                                                                                                                                                           |
| `LogCEI`                       | no                   | LogEI + Σ LogPI (GPs assumed independent)                                                                                                                                                                                                                                        | same                                                                                                                                                                                                                  |
| **`LogEHVI`** (`acqf.py:417`)  | yes                  | see below                                                                                                                                                                                                                                                                        | **univariate Φ/φ only**                                                                                                                                                                                               |
| **`LogCEHVI`**                 | yes, through LogEHVI | ΣLogPI + analytic LogEHVI                                                                                                                                                                                                                                                        | univariate only                                                                                                                                                                                                       |
| `qLogEI`                       | yes                  | qEI (Chevalier & Ginsbourger 2013) over the q-dim joint posterior of (y_running, y_x)                                                                                                                                                                                            | q-variate and (q−1)-variate MVN CDFs, O(q) and O(q²) calls (paper, Lemma 1)                                                                                                                                           |
| `qLogPI`                       | yes                  | the τ→0 limit: 1 − Φ_q(f₀·1; μ, Σ)                                                                                                                                                                                                                                               | one q-variate CDF. Note the current sigmoid with τ = 1e-2 is a different, smoothed quantity                                                                                                                           |
| `qLogCEI`                      | yes                  | E[max_i (y_i−f₀)₊ · 1{i feasible}]. Feasibility is independent of y, so sum over feasible subsets S: P(F=S)·qEI(S). Get P(F=S) by Möbius inversion of P(F⊇T) = Π_c (orthant probability of c over T)                                                                             | 2^q terms, each needing orthant probabilities up to q dims and qEI up to q dims                                                                                                                                       |
| `qLogEHVI`                     | yes                  | Theorem 2: qEHVI = Σ_{I⊆[q]} (−1)^{\|I\|+1} HVI({r̂(I)}, Â(I)), with coordinates from qMinEI per objective                                                                                                                                                                        | 2^q subsets × m objectives × qMinEI, i.e. \|I\|- and (\|I\|−1)-variate CDFs. Optuna's value is E[HVI(y_x \| A ∪ Y_r)] = qEHVI − (q−1)EHVI(running), so you also have to subtract the running-only term (cancellation) |
| `qLogCEHVI`                    | yes                  | E[1{x feasible} · HVI(y_x \| A ∪ feasible Y_r)]. The baseline depends on which running trials turn out feasible, so you need a sum over feasible subsets of running trials, mixed-sign orthant probabilities jointly with x's constraints, and a qEHVI-like term for each subset | exponential in n_running; possible in principle but not practical                                                                                                                                                     |

## LogEHVI (the easy win)

Optuna uses independent GPs per objective (a list GP). That matches the paper's assumption, and `LogEHVI` already builds the box decomposition. With HVI(y) = Σ_k Π_j clamp(y_j − l_kj, 0, Δ_kj), independence across objectives lets the expectation factor per box:

- EHVI = Σ_k Π_j [EI_j(l_kj) − EI_j(u_kj)], where EI_j(a) = E[(y_j − a)₊].
- In log space: log EHVI = logsumexp_k Σ_j log(EI_j(l_kj) − EI_j(u_kj)).

This is the Yang et al. 2019 / BoTorch analytic EHVI (ψ/ν functions). The paper's Theorem 1 is the same result written as HVI({r̃}, Ã), with r̃_j = EI(0, μ_j, Σ_jj) and f̃ⱼ⁽ⁱ⁾ = r̃_j − EI(fⱼ⁽ⁱ⁾, …). That form lets you plug in a fast exact HV algorithm such as QHV-II instead of the box decomposition, which is better for large m but is not autograd-friendly in torch.

The only numerical work is computing log(EI(l) − EI(u)) stably:
- Use `logEI(l) + log1mexp(logEI(u) − logEI(l))`, reusing `standard_logei`.
- When u = +∞ (unbounded boxes), the term is just `logEI(l)`.
- When u − l is tiny, cancellation hurts. Use ∫ₗᵘ Φ((μ−w)/σ) dw ≈ (u−l)·Φ((μ−(l+u)/2)/σ), or a short Gauss–Legendre rule.

## What the batch versions need numerically

**MVN CDF:**
- Dimension 1: `ndtr`/`log_ndtr`.
- Dimension 2: Drezner–Wesolowsky, or Genz 2004 (Gauss–Legendre). Deterministic.
- Dimension 3: Genz 2004 trivariate (Plackett reduction plus 1-D quadrature). Deterministic.
- Dimension ≥ 4: Genz–Bretz (randomized QMC). The deterministic alternatives are a fixed lattice rule, which is still a QMC-style rule, or Miwa–Hayter–Kuriki 2003, which is exact but grows roughly like q! and is usable up to about q ≈ 10–20.

**Gradients:**
- ∂Φ_q/∂b_i = φ(b_i)·Φ_{q−1}(conditional), so it needs a (q−1)-variate CDF.
- Covariance derivatives go through Plackett's identity and need (q−2)-variate CDFs.
- The paper (Corollary 6, citing Marmin et al. 2015) states O(q^{q−k+1}) k-variate CDF calls for q−3 ≤ k ≤ q.
- There is no torch MVN CDF, so you would have to write the forward and backward passes yourself. The paper didn't differentiate qEHVI for this reason.

**Log-space stability:** the inclusion–exclusion sums (signs (−1)^{|I|+1}, Möbius inversion, 1 − Φ_q) cancel badly in log space. That undoes much of what LogEI/LogEHVI were designed to fix.

## Practical middle ground (my inference, not from the paper)

In every q-variant, the candidate given fixed fantasies is Gaussian with variance that doesn't depend on the fantasy: `cond_mean` is linear in the fantasy and `cond_cov` is fantasy-free (`gp.py:410-415`). So you can integrate out only the candidate's QMC dimension and keep QMC only over the fantasies:

- `qLogEI`: per fantasy, compute logEI(max(f₀, max y_r); cond_mean, cond_var), i.e. univariate EI.
- `qLogEHVI`: per fantasy, compute the analytic box-EHVI above against the per-sample baseline that `_PerSampleParetoBaseline` already builds.
- `qLogCEHVI`: per fantasy, add Σ log Φ(conditional constraint) to the analytic EHVI.

This needs only univariate Φ/φ and lowers variance. It is fully QMC-free when n_running = 0. When n_running = 1, Gauss–Hermite quadrature over the single fantasy dimension gives a deterministic, QMC-free result.

`qLogPI` and `qLogCEI` take a max over the whole batch, so the fantasy's own utility stays inside the max. They still split the same way, though: given a fantasy, the max becomes a univariate EI or Φ term at threshold max(f₀, fantasies).