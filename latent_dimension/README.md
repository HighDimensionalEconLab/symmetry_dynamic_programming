# Latent dimension of a DeepSet for conditional aggregation

How wide must a DeepSet's pooled latent be to learn a population aggregate, when the cross-sectional
distribution is governed by a hidden conditioning state? This directory establishes, empirically:

> **`L ≥ d_eff`** — the DeepSet's latent width `L` must be at least `d_eff`, the dimension of the
> **minimal sufficient statistic** of the latent conditioning state for the target. Crucially,
> `d_eff` can be **smaller** than the number of varying DGP parameters when their effects on the
> target are **collinear** (latent compression), and it is set by the **aggregator's moment
> sensitivity**, not by the raw parameter count.

All experiments import the standalone trainer `../generalized_mean_simple.py` and run on a single GPU.
Reference outputs from the runs discussed below are in `results/`. To regenerate: `bash run_all.sh`.

---

## 1. Setup

**Model.** A permutation-invariant DeepSet: per-element `φ` → mean-pool over the set → `ρ`.
`L` is the pooled (latent) width — the number of summary statistics the network may carry.
`L = 0` is a valid degenerate case: the pooled feature is 0-dimensional, so the net emits a learned
**constant** (the unconditional-mean predictor).

**Target.** Generalized ("power") mean of the absolute values of an `N`-vector,

```
Y = ( (1/N) Σ_i |x_i|^p )^(1/p),     p = 1.5.
```

**DGP.** Each set is `N` i.i.d. draws from a sinh-arcsinh (SHASH) shock with a per-set latent
`θ = (μ, σ, ε, δ)`:

```
x_i = μ + σ · sinh( (asinh(z_i) + ε) / δ ),   z_i ~ N(0,1).
```

`μ` = location, `σ` = scale, `ε` = skew, `δ` = tail weight (`ε=0, δ=1` ⇒ Normal). Each parameter is
drawn per set from `Beta(α, α)` scaled to `[min, max]` (`α=1` uniform). The **latent dimension**
`d` is the number of parameters that actually vary (have `min ≠ max`).

---

## 2. Experiment 1 — the latent ladder

Sweep the number of varying parameters `d ∈ {0,1,2,3}` against `L ∈ {0,1,2,3,4,8}` and `N ∈ {128,1024}`
(1000 train / 3000 test points, 3 seeds, median `test_rel_error`). Script: `latent_ladder.py`,
results `results/latent_ladder_results.csv`.

**d = 0 — `const` (nothing varies)**

| L | N=128 | N=1024 |
|---|-------|--------|
| **0** | **1.04e-2** | **3.60e-3** |
| 1 | 3.9e-4 | 4.0e-4 |
| ≥2 | ~3.9e-4 | ~3.9e-4 |

`L=0` is the **correct** rung and it **concentrates** (1.0e-2 → 3.6e-3, the ~1/√N law): with nothing
to condition on, the unconditional mean is optimal and sharpens with `N`.

**d = 1 — `mu`**

| L | N=128 | N=1024 |
|---|-------|--------|
| **0** | 2.88e-1 | 2.88e-1 |
| ≥1 | 7.2e-4 | 4.4e-4 |

`L=0` is **walled at 0.29** — a constant cannot track μ. One latent dimension clears it.

**d = 2 — `mu+sig`** (the cleanest wall)

| L | N=128 | N=1024 |
|---|-------|--------|
| **0** | 2.88e-1 | 2.87e-1 |
| **1** | **3.81e-3** | **4.76e-3** |
| ≥2 | 4.3e-4 | 4.0e-4 |

`L=1` is stuck at ~4e-3 and **N does not help** (drops σ, a strong dimension). `L≥2` needed.

**d = 3 — `mu+sig+eps`**

| L | N=128 | N=1024 |
|---|-------|--------|
| **0** | 3.15e-1 | 3.15e-1 |
| 1 | 5.94e-3 | 6.46e-3 |
| 2 | 3.73e-3 | 5.08e-4 |
| ≥3 | ~4.9e-4 | ~5e-4 |

**Takeaways.** `L=0` is the unconditional-mean baseline (walled when `d≥1`, concentrating when `d=0`);
`L=1` walls for `d≥2`; and large-`N` concentration only helps once `L≥d`. But the `d=3` row already
hints at trouble: `L=2` reaches the floor at `N=1024`. That is the subject of §3.

---

## 3. Experiment 2 — `d=3` "compression": three parameters that behave like two

Deeper nets (3-layer φ/ρ), `N ∈ {64,128,256,1024}`, longer budget (ep15000/600s), so a plateau is a
**capacity** result, not under-training. Script: `d3_compression.py`, results
`results/d3_compression_results.csv`.

| L | N=64 | N=128 | N=256 | N=1024 |
|---|------|-------|-------|--------|
| 1 | 6.9e-3 | 7.5e-3 | 6.5e-3 | 5.9e-3 |
| **2** | **4.1e-4** | **5.2e-4** | **4.3e-4** | **4.0e-4** |
| 3 | 3.9e-4 | 4.4e-4 | 4.5e-4 | 4.2e-4 |
| 4 | 5.3e-4 | 4.6e-4 | 4.5e-4 | 4.7e-4 |
| 8 | 5.5e-4 | 4.5e-4 | 4.5e-4 | 5.1e-4 |

**`L=2` floors at every `N`, even though three parameters vary.** The earlier "`L=2` walled at small
N" was an optimization artifact; with enough budget `L=2` solves `d=3` at all `N`. So the naive law
`L ≥ #parameters` is **wrong** — the truth is `L ≥ d_eff`.

### Why: ε is collinear with location/scale (`collinearity_mc.py`)

The power-mean of a **positive** `X` is a smooth, low-moment functional. Its Taylor expansion is

```
Y ≈ E[X]  +  c₁ · Var[X]/E[X]  +  c₂ · (3rd central moment)  + …
     └loc┘    └──── scale ────┘    └──── skew: tiny ────┘
```

and the SHASH mean shifts with skew, `E[X] = μ + σ·sinh(ε)·E√(1+Z²)`, so **raising ε mostly just
moves the mean** — i.e. ε's effect is collinear with μ (and the remainder with σ). Monte-Carlo,
decomposing the relative spread of `Y` as ε sweeps `[−e,e]` while we strip what a low-dim summary
absorbs (`raw` → fix `E[X]` → fix `E[X]` *and* `Var[X]`):

| ε range | raw (μ,σ fixed) | − location | − location & scale |
|---|---|---|---|
| ±1 | **46%** | 1.0e-2 | **3.4e-4** |
| ±2 | 97% | 11.5% | 4.1e-4 |

ε swings `Y` enormously (46% at ±1 — bigger than σ's entire range), but **~99.9% of that is
location+scale**; the part orthogonal to both is **3.4e-4 — at the learned floor.** So `Y` depends
on only ~2 effective coordinates (location, scale), and the moment hierarchy maps onto the ladder:

| effective summary | orthogonal residual (MC) | matching learned floor |
|---|---|---|
| location only | ~1.0e-2 | `L=1` wall (~6e-3) |
| location + scale | 3.4e-4 | `L=2` floor (~4e-4) |
| + skew (3rd) | nothing left | `L=3` buys nothing |

**`L` tracks the dimension of the sufficient statistic for `Y`, not the parameter count.** Collinear
latents compress. (In a *folding* regime where μ→0 so `X` straddles 0 and `|·|` bites, ε's orthogonal
content jumps ~100× to 3.4e-2 — but that changes the μ range, which we keep fixed here.)

---

## 4. Making ε a genuine 3rd dimension — and choosing the detector

To force a real `d=3` **with μ kept in `[1,3]`**, add a term that depends on ε *orthogonally* to
location and scale:

```
Y = ( (1/N) Σ |x_i|^p )^(1/p)  +  skew_weight · (location/scale-invariant skew statistic).
```

This is implemented as `skew_weight` on `DataSettings` (default 0 → pure power-mean, bit-for-bit).
The library uses **standardized sample skewness** `γ̂ = m₃ / m₂^{3/2}`, which for SHASH is a pure
function of ε (identical across all μ,σ). The detector must be **odd in ε** (to read skew *direction*)
and **location/scale-invariant**. `detector_mc.py` compares candidates by orthogonal content and by
shape:

| detector | orthogonal content (`hold E[X]&Var`) | shape in ε | verdict |
|---|---|---|---|
| standardized skewness `m₃/m₂^{3/2}` | full | odd, monotone | **clean** |
| signed semivariance `E[c\|c\|]/E[c²]` | full (σ²-cancels exactly) | odd, monotone | **clean** |
| Gini `E\|X−X'\|/2E[X]` | 4.6e-2 | **even** (blind to direction) | rejected |
| top/bottom quintile-mean ratio | 4.0e-2 | non-monotone, noisy | muddy |

**Gini is structurally skew-blind.** `E|X−X'|` is reflection-invariant, so after controlling mean and
variance Gini is an *even* function of ε — it gives the *same* value at ε=−1 and ε=+1 and cannot tell
left from right skew. Any pure inequality/dispersion measure (variance, MAD, entropy) shares this:
they see the *magnitude* of spread, never its *direction*. Detecting skew requires an **odd**
functional.

**Skewness vs semivariance.** The semivariance ratio is the second-order analog
`R = (1+η)/(1−η)`, `η = E[c|c|]/E[c²]` (`c = x − x̄`); it is *exactly* μ,σ-invariant (the σ² cancels in
the ratio) and odd in ε, with `R(−ε)=1/R(ε)`. We expected its lower moment order to make it
lower-variance, but the finite-N MC shows γ and η have **the same relative sampling noise** (~0.09 at
N=128) — they are statistically interchangeable for δ=1. η's advantages (boundedness, existence with
only 2nd moments) only matter under heavy tails (varying δ). So the library detector is standardized
skewness; the trajectories below confirm η behaves identically.

---

## 5. Experiment 3 — the skew-augmented `d=3` target

Grid sweep `skew_weight ∈ {0.0, 0.04}` × the Exp-2 `L`/`N` (3-layer nets, ep15000/600s). Script:
`d3_skew.py`, results `results/d3_skew_results.csv`.

`skew_weight=0.0` reproduces the compression table (`L≥2` floors) exactly — a sanity check that the
weight-0 path is inert. `skew_weight=0.04` (median `test_rel_error`):

| L | N=64 | N=128 | N=256 | N=1024 |
|---|------|-------|-------|--------|
| 1 | 1.47e-2 | 1.45e-2 | 1.39e-2 | 1.37e-2 |
| 2 | 1.35e-2 | 1.15e-2 | 1.05e-2 | 9.93e-3 |
| 3 | 1.09e-2 | 9.20e-3 | 8.28e-3 | 5.86e-3 |
| 4 | 8.58e-3 | 6.07e-3 | 5.95e-3 | 4.36e-3 |
| 8 | 5.29e-3 | 5.64e-3 | 6.18e-3 | 3.51e-3 |

**Caveat — this grid under-trains.** *Every* `skew_weight=0.04` run hit `max_epochs` (0/60 converged),
because the skew target trains far slower than the power-mean (stopping is on train-loss ≤ 1e-6, never
reached in 15k epochs). These numbers are a mid-descent snapshot, not capacity floors. The definitive
read needs longer training with periodic evaluation — §6.

---

## 6. Trajectory diagnostics — capacity wall vs optimization polish

Single `N=1024`, `L ∈ {2,3,4}`, `test_rel_error` printed periodically. Script: `skew_trajectory.py`;
logs in `results/trajectory_*.log`. Plateau values:

| L | (a) skewness, uniform ε | (b) semivariance, uniform ε | (c) semivariance, **U-shaped ε** (α=0.5) |
|---|---|---|---|
| 2 | 8.7e-3 | 9.0e-3 | **11.6e-3** (tight plateau) |
| 3 | 5.8e-3 | 6.0e-3 | ~5.8e-3 (bounces 5.5–7.2e-3) |
| 4 | 3.5e-3 | 3.8e-3 | ~4.0e-3 (bounces 3.5–5.5e-3) |

Three findings:

1. **It is not under-training.** Each `L` plateaus by ~15–30k epochs; more epochs only add noise.
   (This is why `d3_skew.py`'s 15k-epoch grid was inconclusive.)

2. **The detector form is not the bottleneck.** Skewness (a) and semivariance (b) — third- vs
   second-order — give **identical** curves. The difficulty is fundamental: the skew signal is a
   *tiny perturbation on large, location-dominated pooled moments* (`E[x²]≈4`, `E[x³]≈8`, moved <1%
   by ε), so the net must resolve those moments to ~0.1% relative precision to extract the 3rd
   coordinate — hard at `N≤1024`, regardless of how the asymmetry is written. There is **no clean
   `L=3` floor**; instead a soft staircase ~10× above the `skew_weight=0` floor.

3. **Capacity wall vs optimization polish.** The robust signal is a sharp **`L=2` capacity wall**
   sitting above an **`L≥3` band**: with two pooled coordinates the optimizer spends them on the
   dominant location+scale (~98% of `Y`) and *structurally cannot* also hold the orthogonal skew
   coordinate, so `L=2`'s error ≈ the unexplained skew variance. The `L=3`→`L=4` gains, by contrast,
   are **over-parameterization easing optimization** — extra degrees of freedom smooth the landscape
   — *not* evidence of `d>3` (at this precision `L=3` and `L=4` overlap in their bounce). **Read the
   intrinsic dimension off where the wall is (`L=2`→`L≥3`), not off marginal gains past it.**

**U-shaped ε** (`Beta(0.5,0.5)`, piling mass at the ±1 corners) does exactly what theory predicts:
it strengthens the skew signal, **lifting the `L=2` wall** (9.0 → 11.6e-3) while leaving the `L≥3`
band put — roughly **doubling** the `L=2`→`L=3` separation (−33% → −53%). It sharpens the **wall**,
not the floor (the learnability ceiling is unchanged).

---

## 7. Conclusions

- **`L ≥ d_eff`**, the dimension of the latent's minimal sufficient statistic for the target — read it
  off the **capacity wall** (where adding a latent dimension produces a discrete drop), not off the
  smooth optimization gains that always continue past it.
- **`d_eff ≤ #parameters`, with equality only when parameter effects are non-collinear.** For a
  power-mean of positive `X`, the skew parameter ε is collinear with the location/scale (mean/variance)
  of the distribution, so three parameters compress to `d_eff = 2` — `L=2` suffices. This collinearity
  of the higher- with the lower-moment parameters *in the target* is exactly why a plain power-mean
  cannot exhibit a genuine 3rd dimension, and why we add an explicit, orthogonal skew term.
- **`d_eff` is a property of the aggregator's moment-sensitivity, not the distribution.** A power-mean
  is skew-blind (`d_eff=2`); a skew-aware aggregate (power-mean + `skew_weight·γ̂`) is genuinely 3D —
  `L=2` walls, `L≥3` clears it — though the weak, location-swamped skew signal does not floor cleanly
  at tractable `N` (a concentration limit, consistent with "`L≥d`, concentration helps once `L≥d`").
- **The right skew detector is an odd, location/scale-invariant moment** (standardized skewness, used
  by the library; or the signed semivariance ratio — interchangeable). Inequality measures (Gini,
  quantile ratios) are reflection-invariant and therefore skew-blind.

---

## 8. Files

| file | what it does |
|---|---|
| `latent_ladder.py` | Exp 1 — `d∈{0..3}` × `L∈{0,1,2,3,4,8}` × `N∈{128,1024}` ladder (incl. `L=0` baseline). |
| `d3_compression.py` | Exp 2 — `d=3` deeper nets; shows `L=2` floors (compression). |
| `d3_skew.py` | Exp 3 — skew-augmented grid, `skew_weight∈{0,0.04}` (note: under-trains at ep15000). |
| `skew_trajectory.py` | Diagnostic — single `N`, per-`L` training trajectory; CLI flags for detector / λ / ε-α / epochs. |
| `collinearity_mc.py` | MC — ε's effect on `Y` decomposes into location+scale; orthogonal residual ~3e-4. |
| `detector_mc.py` | MC — detector comparison (skewness, semivariance, Gini, quantile) + finite-N noise. |
| `run_all.sh` | Runner with every command and parameter variant used. |
| `results/` | Reference CSVs and trajectory logs from the runs above. |

The trainer itself is `../generalized_mean_simple.py` (`skew_weight` on `DataSettings`; CLI via
`jsonargparse`). The grid scripts are resumable (append to a CSV, skip done combos).
