import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
import jsonargparse


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


@dataclass
class DataSettings:
    # Per-row mean a and std are drawn from a symmetric Beta(alpha, alpha) on [min, max].
    # alpha = 1 is uniform, > 1 is hump-shaped (mass to the centre), < 1 is U-shaped (mass to edges).
    # Test bounds/alphas default to the train ones; change them to probe out-of-distribution draws.
    num_train_points: int = 10
    num_test_points: int = 200
    shuffle: bool = True  # one-time shuffle of the training data (rows are i.i.d. either way)
    drop_last: bool = False  # drop the last partial training batch
    train_data_seed: int = 212
    test_data_seed: int = 441
    a_min_train: float = 1.0
    a_max_train: float = 3.0
    a_alpha_train: float = 1.0
    a_min_test: float = 1.0
    a_max_test: float = 3.0
    a_alpha_test: float = 1.0
    std_min_train: float = 0.3
    std_max_train: float = 0.3  # == std_min_train -> fixed std (no variation) by default
    std_alpha_train: float = 1.0
    std_min_test: float = 0.3
    std_max_test: float = 0.3
    std_alpha_test: float = 1.0


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


def simulate_data(num_points, a_min, a_max, a_alpha, std_min, std_max, std_alpha, N, p, seed):
    # Beta.sample() has no generator argument, so isolate and seed the global RNG for the draw.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        a = a_min + (a_max - a_min) * Beta(a_alpha, a_alpha).sample((num_points, 1))  # per-row mean
        std = std_min + (std_max - std_min) * Beta(std_alpha, std_alpha).sample((num_points, 1))  # per-row std
        X = (a + std * torch.randn(num_points, N)).abs()  # rarely negative
    Y = X.pow(p).mean(dim=1).pow(1 / p)  # generalized mean over each row
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
    verbose: bool = True,
):
    assert len({seed, data_set.train_data_seed, data_set.test_data_seed}) == 3, (
        "seed, train_data_seed, test_data_seed must be distinct"
    )

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

    # Generate train and test data.
    X_train, Y_train = simulate_data(
        data_set.num_train_points, data_set.a_min_train, data_set.a_max_train, data_set.a_alpha_train,
        data_set.std_min_train, data_set.std_max_train, data_set.std_alpha_train, N, p,
        data_set.train_data_seed,
    )
    X_test, Y_test = simulate_data(
        data_set.num_test_points, data_set.a_min_test, data_set.a_max_test, data_set.a_alpha_test,
        data_set.std_min_test, data_set.std_max_test, data_set.std_alpha_test, N, p,
        data_set.test_data_seed,
    )

    # The model and data are built on the CPU above; move them to the device once, here.
    model = model.to(device)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_set.lr)
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

        if verbose and opt_set.print_interval > 0 and epoch % opt_set.print_interval == 0:
            print(f"epoch {epoch:4d}  train_loss {train_loss:.3e}")

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
        "a_min_train": data_set.a_min_train,
        "a_max_train": data_set.a_max_train,
        "a_alpha_train": data_set.a_alpha_train,
        "a_min_test": data_set.a_min_test,
        "a_max_test": data_set.a_max_test,
        "a_alpha_test": data_set.a_alpha_test,
        "std_min_train": data_set.std_min_train,
        "std_max_train": data_set.std_max_train,
        "std_alpha_train": data_set.std_alpha_train,
        "std_min_test": data_set.std_min_test,
        "std_max_test": data_set.std_max_test,
        "std_alpha_test": data_set.std_alpha_test,
        "L": hc_set.L,
        "phi_layers": hc_set.phi_layers,
        "phi_hidden_dim": hc_set.phi_hidden_dim,
        "rho_layers": hc_set.rho_layers,
        "rho_hidden_dim": hc_set.rho_hidden_dim,
        "lr": opt_set.lr,
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
    if verbose:
        print(f"stopping_reason={stopping_reason}  epochs_run={epoch + 1}  train_time={train_time:.2f}s")
        print(f"test_loss={test_loss:.3e}  test_rel_error={test_rel_error:.3e}  test_abs_error={test_abs_error:.3e}")
        print(f"results written to {output_file}")

    return results


if __name__ == "__main__":
    jsonargparse.CLI(generalized_mean_simple)
