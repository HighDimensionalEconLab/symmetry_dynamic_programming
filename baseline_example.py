import pandas as pd
import torch
import pytorch_lightning as pl
import yaml
import itertools
import numpy as np

from torch import nn
from jsonargparse import lazy_instance
from torch.utils.data import DataLoader
from econ_layers.utilities import dict_to_cpu
from pytorch_lightning.utilities.cli import LightningCLI
from pathlib import Path
from copy import deepcopy
from typing import Optional


# Local files and utilities
import symmetry_dp.linear_policy_LQ

# TODO: MOVE TO UTILITIES
def gauss_hermite_1D(N):
    nodes, weights = np.polynomial.hermite.hermgauss(N)
    nodes *= np.sqrt(2.0)
    weights *= 1.0 / np.sqrt(np.pi)
    return nodes, weights


def unit_normal_quadrature(dims):
    l = [gauss_hermite_1D(N) for N in dims]
    nodes, weights = list(map(list, zip(*l)))
    nodes = itertools.product(*nodes)
    weights = itertools.product(*weights)
    weights = map(np.prod, weights)
    return np.array(list(nodes)), np.array(list(weights))


# Does not support \mu != 0.  Hence only 1D quadrature required
# Version with deep sets (i.e., network for both Phi and Rho)
class InvestmentEulerBaseline(pl.LightningModule):
    def __init__(
        self,
        N: int,
        alpha_0: float,
        alpha_1: float,
        beta: float,
        gamma: float,
        sigma: float,
        delta: float,
        eta: float,
        nu: float,
        # parameters for method
        verbose: bool,
        omega_quadrature_nodes: int,
        normalize_shock_vector: bool,
        train_trajectories: int,
        val_trajectories: int,
        test_trajectories: int,
        always_simulate_linear: bool,
        batch_size: int,
        shuffle_training: bool,
        T: int,
        X_0_loc: float,
        X_0_scale: float,
        # settings for deep learning approximation
        phi_layers: int,
        phi_dim: int,
        phi_bias: bool,
        rho_layers: int,
        rho_dim: int,
        rho_bias: bool,
        L: int,
        phi_activator: Optional[nn.Module] = lazy_instance(nn.ReLU),
        rho_activator: Optional[nn.Module] = lazy_instance(nn.ReLU),
    ):
        super().__init__()
        self.save_hyperparameters()

        # Networks for the function representation
        self.rho = nn.Sequential(
            nn.Linear(L, rho_dim, bias=rho_bias),
            deepcopy(rho_activator),
            # Add in rho_layers - 1
            *[
                nn.Sequential(
                    nn.Linear(
                        rho_dim,
                        rho_dim,
                        bias=rho_bias,
                    ),
                    deepcopy(rho_activator),
                )
                for i in range(rho_layers - 1)
            ],
            nn.Linear(rho_dim, 1, bias=True),
        )
        self.phi = nn.Sequential(
            nn.Linear(1, phi_dim, bias=phi_bias),
            deepcopy(phi_activator),
            # Add in phi_layers - 1
            *[
                nn.Sequential(
                    nn.Linear(
                        phi_dim,
                        phi_dim,
                        bias=phi_bias,
                    ),
                    deepcopy(phi_activator),
                )
                for i in range(phi_layers - 1)
            ],
            nn.Linear(phi_dim, L, bias=phi_bias),
        )

        # Solves the LQ problem to find the comparison for the baseline
        # used for comparison as well as simulation of datapoints
        self.H_0, self.H_1 = symmetry_dp.linear_policy_LQ.investment_equilibrium_LQ(1, self.hparams)

        # The "simulation_policy" function starts by using the linear_policy
        # to begin the simulation of X_t grid points.  Swaps out later if always_simulate_linear = False
        self.simulation_policy = self.linear_policy
        # Note: deferring some construction to the "setup" step because it occurs on the GPU rather than CPU

    # Used for evaluating u(X) given the current network
    def forward(self, X):
        num_batches, N = X.shape

        # Apply network with the representation and "mean" pooling
        phi_X = torch.stack(
            [torch.mean(self.phi(X[i, :].reshape([N, 1])), 0) for i in range(num_batches)]
        )
        return self.rho(phi_X)

    # An analytic linear policy for simulation and comparison.  Uses LQ solution
    # Exact if \nu = 1.  Used for generating grid of data, not fitting itself.
    def linear_policy(self, X):
        return self.H_0 + self.H_1 * X.mean(1, keepdim=True)

    # Model definition
    def p(self, X):
        # TODO: later can see if a special case to avoid the power for nu = 1 is helpful
        return self.hparams.alpha_0 - self.hparams.alpha_1 * X.mean(2).pow(self.hparams.nu)

    # model residuals given a set of states
    # TODO: This might be cleaned up, but need to be careful with GPUs
    def model_residuals(self, X):
        u_X = self(X)

        # equation (12) and (13)
        X_primes = torch.stack(
            [
                u_X
                + (1 - self.hparams.delta) * X
                + self.hparams.sigma * self.expectation_shock_vector
                + self.hparams.eta * node[0]
                for node in self.nodes
            ]
        ).type_as(X)

        # p(X') expectation
        p_primes = self.p(X_primes)  # n_quadrature_points by T
        Ep = (p_primes.T @ self.quadrature_weights).type_as(X).reshape(-1, 1)

        Eu = (
            (
                torch.stack(tuple(self(X_primes[i]) for i in range(len(self.quadrature_nodes))))
                .squeeze(2)
                .T
                @ self.weights
            )
            .type_as(X)
            .reshape(-1, 1)
        )

        # Euler equation itself
        residuals = self.hparams.gamma * u_X - self.hparams.beta * (
            Ep + self.hparams.gamma * Eu * (1 - self.hparams.delta)
        )  # equation (14)
        return residuals

    def training_step(self, X, batch_idx):
        residuals = self.model_residuals(X)

        loss = (residuals ** 2).sum() / len(residuals)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, X, batch_idx):
        residuals = self.model_residuals(X)

        loss = (residuals ** 2).sum() / len(residuals)

        self.log("val_loss", loss, prog_bar=True)

        # calculate policy error relative to analytic if linear
        if self.params.nu == 1:
            u_ref = self.linear_policy(X)
            u_rel_error = torch.mean(torch.abs(self(X) - u_ref) / torch.abs(u_ref))
            self.log("val_u_rel_error", u_rel_error, prog_bar=True)
            u_abs_error = torch.mean(torch.abs(self(X) - u_ref))
            self.log("val_u_abs_error", u_abs_error, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Test data includes trajectory number, time, etc.

        X = batch["X"]
        residuals = self.model_residuals(X)
        loss = (residuals ** 2).sum() / len(residuals)

        self.log("test_loss", loss, prog_bar=True)

        # Additional logging results
        if self.params.nu == 1:
            u_linear = self.linear_policy(X)
            u_X = self(X)
            u_rel_error = torch.abs(u_X - u_linear) / torch.abs(u_linear)
            u_abs_error = torch.abs(u_X - u_linear)

            self.test_residuals = pd.concat(
                [
                    self.test_residuals,
                    pd.DataFrame(
                        dict_to_cpu(
                            {
                                "t": batch["t"],
                                "ensemble": batch["ensemble"],
                                "u_hat": u_X,
                                "residual": residuals,
                                "u_reference": u_linear,
                            }
                        )
                    ),
                ]
            )
            # Log comparisons
            self.log("test_u_rel_error", torch.mean(u_rel_error), prog_bar=True)
            self.log("test_u_abs_error", torch.mean(u_abs_error), prog_bar=True)
        else:
            u_X = self(X)
            self.test_residuals_df = pd.concat(
                [
                    self.test_residuals_df,
                    pd.DataFrame(
                        dict_to_cpu(
                            {
                                "t": batch["t"],
                                "ensemble": batch["ensemble"],
                                "u_hat": u_X,
                                "residual": residuals,
                            }
                        )
                    ),
                ]
            )

    ## Data and simulation calculations
    def simulate(self, w, omega):
        # TODO: Get number of trajectories from the aggregate shocks/etc.
        num_trajectories = omega.size[0]
        data = torch.zeros(
            num_trajectories,
            self.hparams.T + 1,
            self.hparams.N,
            device=self.device,
            dtype=self.dtype,
        )

        data[:, 0, :] = self.X_0
        for t in range(0, self.hparams.T):
            data[:, t + 1, :] = (
                self.simulation_policy(data[:, t, :])  # num_ensembles by N
                + (1 - self.hparams.delta) * data[:, t, :]
                + self.hparams.sigma * w[:, t, :]
                + self.hparams.eta * omega[:, t]
            )
        return torch.cat(data.unbind(0))  # or something like that?

    # Simulates all of the data using the state space model
    # At this point, the code is running local to the GPU/etc.
    def setup(self, stage):

        # quadrature for use within the expectation calculations
        quadrature_nodes, quadrature_weights = unit_normal_quadrature(
            (self.hparams.omega_quadrature_nodes,)
        )
        self.quadrature_nodes = torch.tensor(quadrature_nodes, dtype=self.dtype, device=self.device)
        self.quadrature_weights = torch.tensor(
            quadrature_weights, dtype=self.dtype, device=self.device
        )

        # Monte Carlo draw for the expectations, possibly normalizing it
        vec = torch.randn(1, self.hparams.N, device=self.device, dtype=self.dtype)
        self.expectation_shock_vector = (
            (vec - vec.mean()) / vec.std() if self.hparams.normalize_shock_vector else vec
        )

        # Draw initial condition for the X_0 to simulate
        self.X_0_dist = torch.distributions.normal.Normal(  # not a tensor
            self.hparams.X_0_loc, self.hparams.X_0_scale
        )
        self.X_0 = torch.abs(
            self.X_0_dist.sample((self.hparams.N,)),
            device=self.device,
            dtype=self.dtype,
        )

        if stage == "fit" or stage is None:
            # Create shocks for reuse during simulation.  Fixed to prevent too radical of changes during the fitting process, but not especially important
            self.omega_train = torch.randn(
                self.hparams.train_trajectories,
                self.hparams.T,
                1,
                device=self.device,
                dtype=self.dtype,
            )
            self.w_train = torch.randn(
                self.hparams.train_trajectories,
                self.hparams.T,
                self.hparams.N,
                device=self.device,
                dtype=self.dtype,
            )

            self.omega_val = torch.randn(
                self.hparams.val_trajectories,
                self.hparams.T,
                1,
                device=self.device,
                dtype=self.dtype,
            )
            self.w_val = torch.randn(
                self.hparams.val_trajectories,
                self.hparams.T,
                self.hparams.N,
                device=self.device,
                dtype=self.dtype,
            )

            # Simulate fixing the shock sequence
            self.train_data = self.simulate(
                self.w_train,
                self.omega_train,
            )
            self.val_data = self.simulate(
                self.w_val,
                self.omega_val,
            )

            # switch future simulations to use the network?
            if self.hparams.always_simulate_linear is False:
                self.simulation_policy = (
                    self.forward
                )  # use internal neural network.  TODO: Check if forward is correct?

        if stage == "test" or stage is None:

            test_trajectories = self.hparams.test_trajectories

            self.omega_test = torch.randn(
                self.hparams.test_trajectories,
                self.hparams.T,
                1,
                device=self.device,
                dtype=self.dtype,
            )
            self.w_test = torch.randn(
                self.hparams.test_trajectories,
                self.hparams.T,
                self.hparams.N,
                device=self.device,
                dtype=self.dtype,
            )
            self.test_data = self.simulate(
                self.w_test,
                self.omega_test,
            )

            # metadata zipping
            # TODO: VERIFY STRUCTURE
            zipped = [
                {"ensemble": n, "t": t, "X": data[n, t, :]}
                for n in range(test_trajectories)
                for t in range(T + 1)
            ]
            self.test_data = zipped  # used by the dataloader
            self.test_results = pd.DataFrame()

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size
            if self.hparams.batch_size > 0
            else len(self.train_data),
            shuffle=self.hparams.shuffle_training,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size
            if self.hparams.batch_size > 0
            else len(self.val_data),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size
            if self.hparams.batch_size > 0
            else len(self.test_data),
        )


def save_results(trainer, model, metrics_dict, print_metrics=True):
    metrics_path = Path(trainer.log_dir) / "metrics.yaml"
    if print_metrics:
        print(metrics_dict)

    with open(metrics_path, "w") as fp:
        yaml.dump(metrics_dict, fp)

    # Store the test_results field on model if it exists
    if hasattr(model, "test_results"):
        model.test_results.to_csv(Path(trainer.log_dir) / "test_results.csv", index=False)


def cli_main():
    cli = LightningCLI(
        InvestmentEulerBaseline,
        run=False,
        seed_everything_default=123,
        save_config_overwrite=True,
        parser_kwargs={"default_config_files": ["baseline_example_defaults.yaml"]},
    )

    # Fit the model
    cli.trainer.fit(cli.model)
    metrics_dict = dict_to_cpu(cli.trainer.logged_metrics.copy())

    # Generate test data
    cli.trainer.test(cli.model)
    metrics_dict.update(dict_to_cpu(cli.trainer.logged_metrics))

    # print results
    print(cli.model.test_results)

    # save results
    save_results(cli.trainer, cli.model, metrics_dict)


if __name__ == "__main__":
    cli_main()
