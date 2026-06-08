"""Why we needed a skew term -- Monte-Carlo evidence that eps is collinear with location/scale.

Population target  Y(theta) = (E|X|^p)^(1/p),  X = mu + sigma*sinh((asinh(Z)+eps)/delta),  Z~N(0,1).
Large N -> Y concentrates to the population value; common random numbers (one shared Z) make the
eps-differences precise.

The decomposition strips what a low-dim summary can absorb:
  raw           : vary eps, mu & sigma fixed                 -> total eps effect on Y
  hold E[X]     : vary eps, re-solve mu to fix E[X]          -> effect NOT explained by location
  hold E[X]&Var : vary eps, re-solve mu,sigma to fix both    -> effect orthogonal to location+scale
The last column is the genuine 3rd-dimension content of eps.  In the experiment regime (E[X]=2, X>0)
it is ~3e-4 -- at the learned floor -- so eps is ~collinear with mu (location) and the power-mean
target has effective dimension 2.  In a folding regime (E[X]=0.5, X straddles 0) it jumps ~100x.

Run:  uv run python latent_dimension/collinearity_mc.py
"""

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
P, DELTA, N = 1.5, 1.0, 20_000_000

torch.manual_seed(0)
Z = torch.randn(N, device=dev, dtype=DT)       # shared across every (mu,sigma,eps) -> CRN
Z2 = torch.randn(N, device=dev, dtype=DT)       # independent draw, to confirm concentration


def Y(mu, sigma, eps, z=Z):
    X = mu + sigma * torch.sinh((torch.asinh(z) + eps) / DELTA)
    return X.abs().pow(P).mean().pow(1.0 / P).item()


def rel_spread(vals):
    lo, hi = min(vals), max(vals)
    return (hi - lo) / (0.5 * abs(lo + hi))


print(f"device={dev}  N={N:,}  p={P}  delta={DELTA}\n")

# --- 1) Y(eps) curve at the central (mu, sigma); confirm concentration on a couple points ---
mu0, sig0 = 2.0, 0.3
eps_grid = [-2, -1.5, -1, -0.5, 0.0, 0.5, 1, 1.5, 2]
ys = {e: Y(mu0, sig0, e) for e in eps_grid}
print(f"Y(eps) at mu={mu0}, sigma={sig0}  (raw: eps moves Y a LOT):")
for e in eps_grid:
    chk = f"   (seed2: {Y(mu0, sig0, e, Z2):.6f})" if e in (-1.0, 0.0, 1.0) else ""
    print(f"  eps={e:+5.1f}   Y={ys[e]:.6f}{chk}")
print(f"  raw rel-spread of Y over eps[-1,1]: {rel_spread([ys[e] for e in (-1, -0.5, 0, 0.5, 1)]):.2e}")

# --- 2) orthogonal decomposition in two regimes ---
def decompose(EXstar, sig_ref, label):
    Vstar = sig_ref ** 2
    print(f"\n{label}  (sigma_ref={sig_ref} -> Var*={Vstar:.3f}):")
    print(f"  {'eps-range':12} {'raw':>10} {'holdE[X]':>10} {'holdE[X]&Var':>14}")
    for e in (1.0, 2.0):
        grid = [-e, -e / 2, 0.0, e / 2, e]
        raw, hloc, hlocsc = [], [], []
        for ev in grid:
            W = torch.sinh((torch.asinh(Z) + ev) / DELTA)
            mW, vW = W.mean().item(), W.var(unbiased=False).item()
            raw.append(Y(EXstar, sig_ref, ev))                           # mu, sigma fixed
            hloc.append(Y(EXstar - sig_ref * mW, sig_ref, ev))           # E[X] fixed
            sig = (Vstar / vW) ** 0.5
            hlocsc.append(Y(EXstar - sig * mW, sig, ev))                 # E[X] & Var fixed
        print(f"  +/-{e:<9} {rel_spread(raw):10.2e} {rel_spread(hloc):10.2e} {rel_spread(hlocsc):14.2e}")

decompose(2.0, 0.3, "A: E[X]=2.0 (experiment regime, X>0)")
decompose(0.5, 0.3, "B: E[X]=0.5 (folding, X straddles 0)")

# --- 3) calibration: how much do mu / sigma / eps each move Y over their DGP ranges? ---
mu_spread = rel_spread([Y(m, 0.3, 0.0) for m in (1.0, 1.5, 2.0, 2.5, 3.0)])     # mu in [1,3]
sig_spread = rel_spread([Y(2.0, s, 0.0) for s in (0.1, 0.2, 0.3, 0.4, 0.5)])    # sigma in [0.1,0.5]
print("\nCalibration (relative spread of Y over each parameter's DGP range; floor ~4e-4):")
print(f"  mu    in [1,3]     (sigma=0.3, eps=0): {mu_spread:.2e}")
print(f"  sigma in [0.1,0.5] (mu=2,     eps=0): {sig_spread:.2e}")
print(f"  eps   in [-1,1]    (mu=2, sigma=0.3): {rel_spread([ys[e] for e in (-1,-0.5,0,0.5,1)]):.2e}")
print("\nDONE")
