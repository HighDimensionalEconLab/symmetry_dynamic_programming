"""Choosing the skew detector -- Monte-Carlo comparison of candidate location/scale-invariant terms.

To turn eps into a genuine 3rd dimension (keeping mu in [1,3]) we add  lambda * detector(x)  to the
target, where detector must be a pure skew signal -- orthogonal to location AND scale.  This script
compares candidates on three axes:

  (1) ORTHOGONAL CONTENT: vary eps holding E[X] and Var[X] fixed; how much does the detector still
      move?  (raw / hold E[X] / hold E[X]&Var rel-spreads.)  A clean detector keeps all its signal
      in the last column and is MONOTONE in eps.
  (2) SKEW-DIRECTION sensitivity: a valid detector is ODD in eps (distinguishes left vs right skew).
      Gini -- built from E|X-X'|, which is reflection-invariant -- is EVEN in eps and so skew-blind.
  (3) FINITE-N noise: standardized skewness (3rd-order) vs signed semivariance (2nd-order).

Conclusions used in the README: standardized skewness and the semivariance ratio are clean and
statistically interchangeable; Gini and quantile ratios are muddied with variance / blind to skew
direction; the library uses standardized skewness (skew_weight).

Run:  uv run python latent_dimension/detector_mc.py
"""

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
P = 1.5
N_BIG = 5_000_000          # for sort-based Gini/quantile detectors
N_HUGE = 20_000_000        # for moment-based detectors / population values

torch.manual_seed(0)
Zb = torch.randn(N_BIG, device=dev, dtype=DT)
idx = torch.arange(1, N_BIG + 1, device=dev, dtype=DT)


def detectors(X):
    """All candidate skew detectors for one population sample X (last-dim reduction)."""
    c = X - X.mean()
    m2 = (c ** 2).mean()
    gamma = (c ** 3).mean() / m2.pow(1.5)              # standardized skewness (3rd-order)
    eta = (c * c.abs()).mean() / m2                    # signed semivariance ratio (2nd-order)
    Xs, _ = torch.sort(X)
    n = X.numel()
    k = n // 5
    gini = (2.0 * (idx * Xs).sum() / (n * Xs.sum()) - (n + 1) / n)            # needs X>0
    quint = Xs[-k:].mean() / Xs[:k].mean()                                    # top/bottom quintile-mean ratio
    return {"skewness": gamma.item(), "semivariance": eta.item(),
            "gini": gini.item(), "quintile_ratio": quint.item()}


def rel_spread(v):
    lo, hi = min(v), max(v)
    return (hi - lo) / (0.5 * abs(lo + hi))


# --- (1) orthogonal-content decomposition over eps[-1,1] at (E[X]=2, Var=0.3^2) ---
EXstar, sig0 = 2.0, 0.3
Vstar = sig0 ** 2
eps_grid = [-1, -0.5, 0, 0.5, 1]
names = ["skewness", "semivariance", "gini", "quintile_ratio"]
res = {nm: {m: [] for m in ("raw", "holdE[X]", "holdE[X]&Var")} for nm in names}
for eps in eps_grid:
    W = torch.sinh(torch.asinh(Zb) + eps)
    mW, vW = W.mean().item(), W.var(unbiased=False).item()
    sig = (Vstar / vW) ** 0.5
    confs = {"raw": (EXstar, sig0), "holdE[X]": (EXstar - sig0 * mW, sig0),
             "holdE[X]&Var": (EXstar - sig * mW, sig)}
    for m, (mu, s) in confs.items():
        d = detectors(mu + s * W)
        for nm in names:
            res[nm][m].append(d[nm])

print("(1) orthogonal content -- rel-spread over eps[-1,1]; 'holdE[X]&Var' = pure-skew signal:")
print(f"  {'detector':16} {'raw':>10} {'holdE[X]':>10} {'holdE[X]&Var':>14}")
for nm in names:
    print(f"  {nm:16} {rel_spread(res[nm]['raw']):10.2e} {rel_spread(res[nm]['holdE[X]']):10.2e}"
          f" {rel_spread(res[nm]['holdE[X]&Var']):14.2e}")

# --- (2) skew-direction: is the detector odd (good) or even (Gini, skew-blind) in eps? ---
print("\n(2) holdE[X]&Var values vs eps [-1,-.5,0,.5,1]  (odd=detects direction; even=Gini blind):")
for nm in names:
    print(f"  {nm:16}", [f"{v:+.4f}" for v in res[nm]['holdE[X]&Var']])

# --- (3) finite-N sampling noise: standardized skewness vs semivariance (relative to signal range) ---
torch.manual_seed(0)
Zh = torch.randn(N_HUGE, device=dev, dtype=DT)
def pop(eps):
    X = 2.0 + 0.3 * torch.sinh(torch.asinh(Zh) + eps)
    c = X - X.mean()
    m2 = (c ** 2).mean()
    return ((c ** 3).mean() / m2.pow(1.5)).item(), ((c * c.abs()).mean() / m2).item()
g_lo, e_lo = pop(-1.0)
g_hi, e_hi = pop(1.0)
Dg, De = abs(g_hi - g_lo), abs(e_hi - e_lo)
print(f"\n(3) finite-N noise (M sets at eps=0; relative = std / eps-range Dg={Dg:.2f}, De={De:.2f}):")
print(f"  {'N':>6} | {'rel_skew':>10} {'rel_semivar':>12} {'ratio':>7}")
M = 8000
for Nset in (128, 1000):
    torch.manual_seed(1)
    Zs = torch.randn(M, Nset, device=dev, dtype=DT)
    X = 2.0 + 0.3 * torch.sinh(torch.asinh(Zs) + 0.0)
    c = X - X.mean(-1, keepdim=True)
    m2 = (c ** 2).mean(-1)
    sg = ((c ** 3).mean(-1) / m2.pow(1.5)).std().item()
    se = ((c * c.abs()).mean(-1) / m2).std().item()
    print(f"  {Nset:>6} | {sg / Dg:>10.3f} {se / De:>12.3f} {(sg / Dg) / (se / De):>7.2f}")
print("\nDONE")
