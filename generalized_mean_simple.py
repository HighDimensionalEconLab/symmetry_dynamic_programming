# Standalone, dependency-light trainer (torch + jsonargparse) for a permutation-invariant DeepSet that
# learns the generalized ("power") mean Y = (mean|X|^p)^(1/p) of an N-vector whose elements are i.i.d.
# sinh-arcsinh (SHASH) shocks with a per-set latent (mu, sigma, epsilon, delta). Run as a CLI or import
# generalized_mean_simple(...). The latent_dimension/ experiments use it to study how the DeepSet's
# pooled latent width L relates to the dimension of the latent conditioning state.
import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import jsonargparse
import wandb


@dataclass
class HCSettings:  # hypothesis-class (DeepSet) architecture knobs
    L: int = 4  # latent width after the mean-reduction
    phi_layers: int = 2  # depth before the mean-reduction
    phi_hidden_dim: int = 128
    rho_layers: int = 2  # depth after the mean-reduction
    rho_hidden_dim: int = 128


@dataclass
class OptimizerSettings:
    lr: float = 0.005
    max_epochs: int = 1000
    max_time: float = 180.0  # seconds
    batch_size: int = 32  # <=0 -> full batch
    stopping_threshold: float = 1e-6  # stop when epoch train_loss <= this
    print_interval: int = 50  # print progress every this many epochs (<=0 disables)
    lr_schedule: str = "none"  # "none" or "plateau" (ReduceLROnPlateau on train_loss)
    lr_factor: float = 0.5  # plateau: lr *= lr_factor when train_loss stalls
    lr_patience: int = 100  # plateau: epochs with no train_loss improvement before a drop
    min_lr: float = 1e-6  # plateau: lower bound on the learning rate


@dataclass
class DataSettings:
    # Each set element is a sinh-arcsinh (SHASH) shock:
    #   X = mu + sigma*sinh((asinh(Z) + epsilon)/delta),  Z ~ N(0, 1)   (eps=0, delta=1 -> Normal)
    # and the target is Y = (mean |X|^p)^(1/p). The per-row mu (location), sigma (scale),
    # epsilon (skew) and delta (tail weight) are each drawn from a symmetric Beta(alpha, alpha)
    # scaled to [min, max]: alpha=1 is uniform, >1 hump-shaped, <1 U-shaped. Any *_test field left
    # at None inherits the matching *_train value.
    num_train_points: int = 10
    num_test_points: int = 200
    shuffle: bool = True  # one-time shuffle of the training data (rows are i.i.d. either way)
    drop_last: bool = False  # drop the last partial training batch
    train_data_seed: int = 212
    test_data_seed: int = 441
    mu_min_train: float = 1.0
    mu_max_train: float = 3.0
    mu_alpha_train: float = 1.0
    sigma_min_train: float = 0.3
    sigma_max_train: float = 0.3  # == sigma_min_train -> fixed sigma (no variation) by default
    sigma_alpha_train: float = 1.0
    epsilon_min_train: float = 0.0
    epsilon_max_train: float = 0.0  # == epsilon_min_train -> no skew by default
    epsilon_alpha_train: float = 1.0
    delta_min_train: float = 1.0
    delta_max_train: float = 1.0  # == delta_min_train -> normal tails by default (delta > 0)
    delta_alpha_train: float = 1.0
    mu_min_test: float | None = None
    mu_max_test: float | None = None
    mu_alpha_test: float | None = None
    sigma_min_test: float | None = None
    sigma_max_test: float | None = None
    sigma_alpha_test: float | None = None
    epsilon_min_test: float | None = None
    epsilon_max_test: float | None = None
    epsilon_alpha_test: float | None = None
    delta_min_test: float | None = None
    delta_max_test: float | None = None
    delta_alpha_test: float | None = None
    # Target shaping: Y = (mean|X|^p)^(1/p) + skew_weight * standardized-sample-skewness(X).
    # skew_weight=0 -> pure generalized mean. A nonzero weight adds a location/scale-invariant
    # 3rd-moment term, so the target genuinely depends on the skew (epsilon) dimension.
    skew_weight: float = 0.0


class DeepSet(nn.Module):
    def __init__(self, phi, rho):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, X):
        # X is [batch, N]; unsqueeze to [batch, N, 1] so phi broadcasts over the set
        # dimension, average over the set, then apply rho
        phi_X = self.phi(X.unsqueeze(-1)).mean(dim=1)
        return self.rho(phi_X)


def deepsets_HC(L, phi_layers, phi_hidden_dim, rho_layers, rho_hidden_dim):
    # phi carries biases throughout; rho's hidden layers are bias-free; ReLU between layers
    rho_modules = [nn.Linear(L, rho_hidden_dim, bias=False), nn.ReLU()]
    for _ in range(rho_layers - 1):
        rho_modules += [nn.Linear(rho_hidden_dim, rho_hidden_dim, bias=False), nn.ReLU()]
    rho_modules += [nn.Linear(rho_hidden_dim, 1, bias=True)]
    rho = nn.Sequential(*rho_modules)

    phi_modules = [nn.Linear(1, phi_hidden_dim, bias=True), nn.ReLU()]
    for _ in range(phi_layers - 1):
        phi_modules += [nn.Linear(phi_hidden_dim, phi_hidden_dim, bias=True), nn.ReLU()]
    phi_modules += [nn.Linear(phi_hidden_dim, L, bias=True)]
    phi = nn.Sequential(*phi_modules)

    return DeepSet(phi, rho)


def or_default(value, default):
    return default if value is None else value


def simulate_data(num_points, mu, sigma, epsilon, delta, N, p, seed, skew_weight=0.0):
    # mu/sigma/epsilon/delta are (min, max, alpha): row-param ~ min + (max-min)*Beta(alpha, alpha).
    # Beta.sample() has no generator argument, so isolate and seed the global RNG for the draw.
    def draw(bounds):
        lo, hi, alpha = bounds
        return lo + (hi - lo) * Beta(alpha, alpha).sample((num_points, 1))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m, s, e, d = draw(mu), draw(sigma), draw(epsilon), draw(delta)
        Z = torch.randn(num_points, N)
        X = m + s * torch.sinh((torch.asinh(Z) + e) / d)  # sinh-arcsinh (SHASH) shock, signed
    Y = X.abs().pow(p).mean(dim=1).pow(1 / p)  # generalized mean of |X| over each row
    if skew_weight != 0.0:
        c = X - X.mean(dim=1, keepdim=True)
        m2 = c.pow(2).mean(dim=1)
        skew = c.pow(3).mean(dim=1) / m2.pow(1.5)  # standardized sample skewness over each row
        Y = Y + skew_weight * skew
    return X, Y


def batches(*tensors, batch_size, drop_last=False):
    # Slice already-shuffled, same-length tensors into contiguous mini-batches (views, no copy).
    n = tensors[0].shape[0]
    assert all(t.shape[0] == n for t in tensors)
    stop = (n // batch_size) * batch_size if drop_last else n
    for start in range(0, stop, batch_size):
        yield tuple(t[start : start + batch_size] for t in tensors)


def generalized_mean_simple(
    N: int = 256,
    p: float = 1.5,
    hc_set: HCSettings = HCSettings(),
    data_set: DataSettings = DataSettings(),
    opt_set: OptimizerSettings = OptimizerSettings(),
    seed: int = 123,
    use_gpu: bool = False,  # use a CUDA or MPS device if available, else fall back to CPU
    output_file: str = "generalized_mean_simple_results.json",
    wandb_mode: str = "disabled",  # "disabled", "offline", or "online"
    verbose: bool = True,
):
    assert len({seed, data_set.train_data_seed, data_set.test_data_seed}) == 3, (
        "seed, train_data_seed, test_data_seed must be distinct"
    )

    if not wandb_mode == "disabled":
        wandb.init(project="symmetry", mode=wandb_mode)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"device={device}")

    torch.manual_seed(seed)
    model = deepsets_HC(**vars(hc_set))

    # Generate train and test data. Each (min, max, alpha) tuple defines one SHASH parameter's draw;
    # any unset test field inherits the matching train field.
    mu_train = (data_set.mu_min_train, data_set.mu_max_train, data_set.mu_alpha_train)
    sigma_train = (data_set.sigma_min_train, data_set.sigma_max_train, data_set.sigma_alpha_train)
    epsilon_train = (data_set.epsilon_min_train, data_set.epsilon_max_train, data_set.epsilon_alpha_train)
    delta_train = (data_set.delta_min_train, data_set.delta_max_train, data_set.delta_alpha_train)
    mu_test = (or_default(data_set.mu_min_test, data_set.mu_min_train),
               or_default(data_set.mu_max_test, data_set.mu_max_train),
               or_default(data_set.mu_alpha_test, data_set.mu_alpha_train))
    sigma_test = (or_default(data_set.sigma_min_test, data_set.sigma_min_train),
                  or_default(data_set.sigma_max_test, data_set.sigma_max_train),
                  or_default(data_set.sigma_alpha_test, data_set.sigma_alpha_train))
    epsilon_test = (or_default(data_set.epsilon_min_test, data_set.epsilon_min_train),
                    or_default(data_set.epsilon_max_test, data_set.epsilon_max_train),
                    or_default(data_set.epsilon_alpha_test, data_set.epsilon_alpha_train))
    delta_test = (or_default(data_set.delta_min_test, data_set.delta_min_train),
                  or_default(data_set.delta_max_test, data_set.delta_max_train),
                  or_default(data_set.delta_alpha_test, data_set.delta_alpha_train))
    X_train, Y_train = simulate_data(
        data_set.num_train_points, mu_train, sigma_train, epsilon_train, delta_train, N, p,
        data_set.train_data_seed, data_set.skew_weight,
    )
    X_test, Y_test = simulate_data(
        data_set.num_test_points, mu_test, sigma_test, epsilon_test, delta_test, N, p,
        data_set.test_data_seed, data_set.skew_weight,
    )

    # The model and data are built on the CPU above; move them to the device once, here.
    model = model.to(device)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # Setup the optimizer (and an optional plateau learning-rate schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_set.lr)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=opt_set.lr_factor, patience=opt_set.lr_patience, min_lr=opt_set.min_lr
        )
        if opt_set.lr_schedule == "plateau"
        else None
    )
    batch_size = opt_set.batch_size if opt_set.batch_size > 0 else data_set.num_train_points

    # One-time shuffle of the training data; the batch loop below streams it in fixed order
    if data_set.shuffle:
        perm = torch.randperm(data_set.num_train_points, device=device)
        X_train, Y_train = X_train[perm], Y_train[perm]

    # Run optimizer
    start = time.perf_counter()
    train_loss = math.nan
    stopping_reason = "max_epochs"
    for epoch in range(opt_set.max_epochs):
        epoch_sq_error = 0.0
        n_seen = 0
        for X_batch, Y_batch in batches(
            X_train, Y_train, batch_size=batch_size, drop_last=data_set.drop_last
        ):
            # Reset gradients and execute primal
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(X_batch), Y_batch.unsqueeze(1))

            # Run AD and finish optimizer step
            loss.backward()
            optimizer.step()
            epoch_sq_error += loss.item() * len(X_batch)
            n_seen += len(X_batch)
        train_loss = epoch_sq_error / n_seen  # n_seen handles drop_last
        if scheduler is not None:
            scheduler.step(train_loss)

        if verbose and opt_set.print_interval > 0 and epoch % opt_set.print_interval == 0:
            print(f"epoch {epoch:4d}  train_loss {train_loss:.3e}  lr {optimizer.param_groups[0]['lr']:.2e}")

        if train_loss <= opt_set.stopping_threshold:
            stopping_reason = "stopping_threshold"
            break
        if time.perf_counter() - start > opt_set.max_time:
            stopping_reason = "max_time"
            break
    train_time = time.perf_counter() - start

    # Evaluate on the test set (accumulate over batches -> same as a single-pass mean)
    model.eval()
    test_sq_sum = test_abs_sum = test_rel_sum = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in batches(X_test, Y_test, batch_size=batch_size):
            test_pred = model(X_batch)
            test_target = Y_batch.unsqueeze(1)
            test_residuals = test_target - test_pred
            abs_res = test_residuals.abs()
            test_sq_sum += (test_residuals ** 2).sum().item()
            test_abs_sum += abs_res.sum().item()
            test_rel_sum += (abs_res / test_target.abs()).sum().item()
    test_loss = test_sq_sum / data_set.num_test_points
    test_abs_error = test_abs_sum / data_set.num_test_points
    test_rel_error = test_rel_sum / data_set.num_test_points

    total_params = sum(p_.numel() for p_ in model.parameters())
    trainable_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)

    results = {
        "test_loss": test_loss,
        "test_rel_error": test_rel_error,
        "test_abs_error": test_abs_error,
        "train_loss": train_loss,
        "epochs_run": epoch + 1,
        "stopping_reason": stopping_reason,
        "train_time": train_time,
        "device": str(device),
        "total_params": total_params,
        "trainable_params": trainable_params,
        # resolved config
        "N": N,
        "p": p,
        "mu_min_train": data_set.mu_min_train,
        "mu_max_train": data_set.mu_max_train,
        "mu_alpha_train": data_set.mu_alpha_train,
        "mu_min_test": mu_test[0],
        "mu_max_test": mu_test[1],
        "mu_alpha_test": mu_test[2],
        "sigma_min_train": data_set.sigma_min_train,
        "sigma_max_train": data_set.sigma_max_train,
        "sigma_alpha_train": data_set.sigma_alpha_train,
        "sigma_min_test": sigma_test[0],
        "sigma_max_test": sigma_test[1],
        "sigma_alpha_test": sigma_test[2],
        "epsilon_min_train": data_set.epsilon_min_train,
        "epsilon_max_train": data_set.epsilon_max_train,
        "epsilon_alpha_train": data_set.epsilon_alpha_train,
        "epsilon_min_test": epsilon_test[0],
        "epsilon_max_test": epsilon_test[1],
        "epsilon_alpha_test": epsilon_test[2],
        "delta_min_train": data_set.delta_min_train,
        "delta_max_train": data_set.delta_max_train,
        "delta_alpha_train": data_set.delta_alpha_train,
        "delta_min_test": delta_test[0],
        "delta_max_test": delta_test[1],
        "delta_alpha_test": delta_test[2],
        "skew_weight": data_set.skew_weight,
        "L": hc_set.L,
        "phi_layers": hc_set.phi_layers,
        "phi_hidden_dim": hc_set.phi_hidden_dim,
        "rho_layers": hc_set.rho_layers,
        "rho_hidden_dim": hc_set.rho_hidden_dim,
        "lr": opt_set.lr,
        "lr_schedule": opt_set.lr_schedule,
        "lr_factor": opt_set.lr_factor,
        "lr_patience": opt_set.lr_patience,
        "min_lr": opt_set.min_lr,
        "final_lr": optimizer.param_groups[0]["lr"],
        "batch_size": opt_set.batch_size,
        "max_epochs": opt_set.max_epochs,
        "max_time": opt_set.max_time,
        "stopping_threshold": opt_set.stopping_threshold,
        "num_train_points": data_set.num_train_points,
        "num_test_points": data_set.num_test_points,
        "shuffle": data_set.shuffle,
        "drop_last": data_set.drop_last,
        "seed": seed,
        "train_data_seed": data_set.train_data_seed,
        "test_data_seed": data_set.test_data_seed,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    if not wandb_mode == "disabled":
        wandb.log(results)
        wandb.finish()

    if verbose:
        print(f"stopping_reason={stopping_reason}  epochs_run={epoch + 1}  train_time={train_time:.2f}s")
        print(f"test_loss={test_loss:.3e}  test_rel_error={test_rel_error:.3e}  test_abs_error={test_abs_error:.3e}")
        print(f"results written to {output_file}")

    return results


if __name__ == "__main__":
    jsonargparse.CLI(generalized_mean_simple)
