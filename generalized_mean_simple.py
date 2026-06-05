import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import jsonargparse


@dataclass
class HCSettings:  # hypothesis-class (DeepSet) architecture knobs
    L: int = 2  # latent width after the mean-reduction
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
    num_train_points: int = 10
    num_val_points: int = 50
    num_test_points: int = 200
    train_data_seed: int = 0  # >0 uses a dedicated RNG; 0 uses the global RNG
    test_seed: int = 0


class DeepSet(nn.Module):
    def __init__(self, phi, rho):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, X):
        num_batches, N = X.shape
        # phi on every set element, mean-reduce over the set, then rho
        phi_X = self.phi(X.reshape(num_batches * N, 1)).reshape(num_batches, N, -1).mean(dim=1)
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


def simulate_data(num_points, a_min, a_max, std, X_distribution, N, p, generator=None):
    X = torch.empty(num_points, N)
    for i in range(num_points):
        a_i = a_min + (a_max - a_min) * torch.rand(1, generator=generator).item()
        if X_distribution == "normal":
            X[i] = torch.normal(a_i, std, size=(N,), generator=generator).abs()  # rarely negative
        elif X_distribution == "uniform":
            d = std * math.sqrt(3)  # ensures std is correct
            X[i] = torch.rand(N, generator=generator) * 2 * d + a_i - d
        else:
            raise ValueError("Distribution not supported")
    Y = X.pow(p).mean(dim=1).pow(1 / p)  # generalized mean over each row
    return X, Y


def regression_test():
    seed = 42

    # Point generation, normal branch
    gen = torch.Generator().manual_seed(seed)
    X, Y = simulate_data(3, 1.0, 3.0, 0.3, "normal", 8, 1.5, generator=gen)
    X_expected = torch.tensor(
        [
            [2.57309222, 3.17116976, 2.86226416, 2.50425005],
            [2.26571798, 2.27761960, 1.73899567, 2.55474091],
            [1.46080899, 1.71370208, 1.62720478, 1.01046097],
        ]
    )
    Y_expected = torch.tensor([2.81091690, 2.16113520, 1.47641218])
    assert torch.allclose(X[:, :4], X_expected, atol=1e-5), X[:, :4]
    assert torch.allclose(Y, Y_expected, atol=1e-5), Y

    # Point generation, uniform branch
    gen_u = torch.Generator().manual_seed(seed)
    _, Yu = simulate_data(3, 1.0, 3.0, 0.3, "uniform", 8, 1.5, generator=gen_u)
    Yu_expected = torch.tensor([2.93211579, 1.47903740, 1.43797660])
    assert torch.allclose(Yu, Yu_expected, atol=1e-5), Yu

    # Residual calculation: default-architecture DeepSet applied to the N=8 fixture
    torch.manual_seed(seed)
    model = deepsets_HC(L=2, phi_layers=2, phi_hidden_dim=128, rho_layers=2, rho_hidden_dim=128)
    with torch.no_grad():
        pred = model(X)
    residuals = Y.unsqueeze(1) - pred
    pred_expected = torch.tensor([-0.10764635, -0.10277723, -0.09936349])
    residuals_expected = torch.tensor([2.91856337, 2.26391244, 1.57577562])
    assert torch.allclose(pred.flatten(), pred_expected, atol=1e-5), pred.flatten()
    assert torch.allclose(residuals.flatten(), residuals_expected, atol=1e-5), residuals.flatten()

    print("Regression test PASSED")


def generalized_mean_simple(
    a_min: float = 1.0,
    a_max: float = 3.0,
    std: float = 0.3,
    X_distribution: str = "normal",  # "normal" or "uniform"
    N: int = 256,
    p: float = 1.5,
    hc_set: HCSettings = HCSettings(),
    opt_set: OptimizerSettings = OptimizerSettings(),
    seed: int = 123,
    output_file: str = "generalized_mean_simple_results.json",
    verbose: bool = True,
    run_regression_test: bool = False,
):
    if run_regression_test:
        regression_test()
        return

    torch.manual_seed(seed)
    model = deepsets_HC(
        hc_set.L,
        hc_set.phi_layers,
        hc_set.phi_hidden_dim,
        hc_set.rho_layers,
        hc_set.rho_hidden_dim,
    )

    if opt_set.train_data_seed > 0:
        train_gen = torch.Generator().manual_seed(opt_set.train_data_seed)
    else:
        train_gen = None
    if opt_set.test_seed > 0:
        test_gen = torch.Generator().manual_seed(opt_set.test_seed)
    else:
        test_gen = None

    # Draw the test set before training so its metrics are independent of the training RNG.
    # Val shares the train generator.
    X_train, Y_train = simulate_data(
        opt_set.num_train_points, a_min, a_max, std, X_distribution, N, p, generator=train_gen
    )
    X_val, Y_val = simulate_data(
        opt_set.num_val_points, a_min, a_max, std, X_distribution, N, p, generator=train_gen
    )
    X_test, Y_test = simulate_data(
        opt_set.num_test_points, a_min, a_max, std, X_distribution, N, p, generator=test_gen
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_set.lr)
    batch_size = opt_set.batch_size if opt_set.batch_size > 0 else opt_set.num_train_points

    start = time.perf_counter()
    train_loss = math.nan
    stopping_reason = "max_epochs"
    for epoch in range(opt_set.max_epochs):
        perm = torch.randperm(opt_set.num_train_points)
        epoch_sq_error = 0.0
        for b in range(0, opt_set.num_train_points, batch_size):
            idx = perm[b : b + batch_size]
            optimizer.zero_grad()
            loss = F.mse_loss(model(X_train[idx]), Y_train[idx].unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_sq_error += loss.item() * len(idx)
        train_loss = epoch_sq_error / opt_set.num_train_points

        if verbose and opt_set.print_interval > 0 and epoch % opt_set.print_interval == 0:
            print(f"epoch {epoch:4d}  train_loss {train_loss:.3e}")

        if train_loss <= opt_set.stopping_threshold:
            stopping_reason = "stopping_threshold"
            break
        if time.perf_counter() - start > opt_set.max_time:
            stopping_reason = "max_time"
            break
    train_time = time.perf_counter() - start

    with torch.no_grad():
        val_pred = model(X_val)
        val_true = Y_val.unsqueeze(1)
        val_residuals = val_true - val_pred
        val_loss = F.mse_loss(val_pred, val_true).item()
        val_rel_error = torch.mean(torch.abs(val_residuals) / torch.abs(val_true)).item()
        val_abs_error = torch.mean(torch.abs(val_residuals)).item()

        test_pred = model(X_test)
        test_true = Y_test.unsqueeze(1)
        test_residuals = test_true - test_pred
        test_loss = F.mse_loss(test_pred, test_true).item()
        test_rel_error = torch.mean(torch.abs(test_residuals) / torch.abs(test_true)).item()
        test_abs_error = torch.mean(torch.abs(test_residuals)).item()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "test_loss": test_loss,
        "test_rel_error": test_rel_error,
        "test_abs_error": test_abs_error,
        "val_loss": val_loss,
        "val_rel_error": val_rel_error,
        "val_abs_error": val_abs_error,
        "train_loss": train_loss,
        "epochs_run": epoch + 1,
        "stopping_reason": stopping_reason,
        "train_time": train_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        # resolved config
        "N": N,
        "p": p,
        "a_min": a_min,
        "a_max": a_max,
        "std": std,
        "X_distribution": X_distribution,
        "L": hc_set.L,
        "phi_layers": hc_set.phi_layers,
        "phi_hidden_dim": hc_set.phi_hidden_dim,
        "rho_layers": hc_set.rho_layers,
        "rho_hidden_dim": hc_set.rho_hidden_dim,
        "lr": opt_set.lr,
        "batch_size": opt_set.batch_size,
        "num_train_points": opt_set.num_train_points,
        "num_val_points": opt_set.num_val_points,
        "num_test_points": opt_set.num_test_points,
        "seed": seed,
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
