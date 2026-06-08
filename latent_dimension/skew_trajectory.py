"""Diagnostic -- training trajectory for the skew-augmented target at a single N, several L.

Prints test_rel_error every --eval-interval epochs so we can see (a) WHERE training plateaus (the
d3_skew.py grid hit max_epochs everywhere -> under-trained), and (b) the shape of the L-staircase:
L=2 is a CAPACITY wall (cannot represent the 3rd/skew coordinate while holding the dominant
location+scale), whereas L=3->L=4 gains are OPTIMIZATION polish (redundant degrees of freedom).

Target:  Y = (mean|x|^p)^(1/p) + lambda * detector(x),  detector in:
  skewness     : standardized sample skewness  m3 / m2^{3/2}            (third-order, == library skew_weight)
  semivariance : signed semivariance ratio      E[c|c|] / E[c^2]         (second-order; c = x - mean)
Both are location/scale-invariant pure functions of eps; they are statistically interchangeable
(see detector_mc.py / README). eps ~ Beta(eps_alpha, eps_alpha) on [eps_min,eps_max]: alpha=1 uniform,
alpha=0.5 U-shaped (piles mass at the corners +/-1 -> stronger skew, sharper L=2 wall).

Examples (the three runs behind the README trajectory tables):
  uv run python latent_dimension/skew_trajectory.py --detector skewness     --lam 0.04 --eps-alpha 1.0 --max-epochs 50000 --eval-interval 5000
  uv run python latent_dimension/skew_trajectory.py --detector semivariance --lam 0.14 --eps-alpha 1.0 --max-epochs 50000 --eval-interval 5000
  uv run python latent_dimension/skew_trajectory.py --detector semivariance --lam 0.14 --eps-alpha 0.5 --max-epochs 20000 --eval-interval 2500
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generalized_mean_simple import deepsets_HC, simulate_data, batches  # noqa: E402


def detector_term(X, kind):
    c = X - X.mean(dim=1, keepdim=True)
    m2 = c.pow(2).mean(dim=1)
    if kind == "skewness":
        return c.pow(3).mean(dim=1) / m2.pow(1.5)          # standardized skewness (3rd-order)
    if kind == "semivariance":
        return (c * c.abs()).mean(dim=1) / m2              # signed semivariance ratio (2nd-order)
    raise ValueError(f"unknown detector {kind}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=["skewness", "semivariance"], default="skewness")
    ap.add_argument("--lam", type=float, default=0.04, help="skew-term weight (0 -> pure power-mean)")
    ap.add_argument("--eps-alpha", type=float, default=1.0, help="Beta(a,a) for eps: 1 uniform, 0.5 U-shaped")
    ap.add_argument("--eps-min", type=float, default=-1.0)
    ap.add_argument("--eps-max", type=float, default=1.0)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--L", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--p", type=float, default=1.5)
    ap.add_argument("--max-epochs", type=int, default=20000)
    ap.add_argument("--eval-interval", type=int, default=2500)
    ap.add_argument("--num-train", type=int, default=1000)
    ap.add_argument("--num-test", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--phi-layers", type=int, default=3)
    ap.add_argument("--rho-layers", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sigma, delta = (1.0, 3.0, 1.0), (0.1, 0.5, 1.0), (1.0, 1.0, 1.0)  # mu,sigma fixed-range
    eps = (a.eps_min, a.eps_max, a.eps_alpha)

    def make(num, seed):
        # base power-mean target (skew_weight=0), then add lam * detector computed from the same X
        X, Ybase = simulate_data(num, mu, sigma, eps, delta, a.N, a.p, seed, 0.0)
        return X.to(dev), (Ybase + a.lam * detector_term(X, a.detector)).to(dev)

    Xtr, Ytr = make(a.num_train, a.seed + 100)
    Xte, Yte = make(a.num_test, a.seed + 200)
    Yte_u = Yte.unsqueeze(1)

    def test_rel(model):
        model.eval()
        with torch.no_grad():
            rel = ((Yte_u - model(Xte)).abs() / Yte_u.abs()).mean().item()
        model.train()
        return rel

    print(f"device={dev}  detector={a.detector}  lambda={a.lam}  N={a.N}  "
          f"eps~Beta({a.eps_alpha},{a.eps_alpha}) on [{a.eps_min},{a.eps_max}]  "
          f"({a.max_epochs} epochs, test_rel every {a.eval_interval})", flush=True)
    for L in a.L:
        torch.manual_seed(a.seed)
        model = deepsets_HC(L=L, phi_layers=a.phi_layers, phi_hidden_dim=a.hidden,
                            rho_layers=a.rho_layers, rho_hidden_dim=a.hidden).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        perm = torch.randperm(a.num_train, device=dev)
        Xs, Ys = Xtr[perm], Ytr[perm]
        print(f"--- L={L} ---", flush=True)
        t0 = time.perf_counter()
        for epoch in range(a.max_epochs + 1):
            for xb, yb in batches(Xs, Ys, batch_size=a.batch_size):
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(xb), yb.unsqueeze(1))
                loss.backward()
                opt.step()
            if epoch % a.eval_interval == 0:
                print(f"  epoch {epoch:6d}  train_mse {loss.item():.2e}  test_rel {test_rel(model):.3e}"
                      f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
