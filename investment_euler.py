import pandas as pd
import torch
import pytorch_lightning as pl
import yaml
import math
import itertools
import numpy as np
import wandb
import warnings
import timeit
import quantecon
import econ_layers

from scipy import optimize
from torch import nn
from torch.utils.data import DataLoader
from econ_layers.utilities import dict_to_cpu
from pytorch_lightning.cli import LightningCLI
from pathlib import Path
from copy import deepcopy
from typing import Optional
from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings(
    action="ignore", category=UserWarning, message="Due to class_path change from"
)

# Version with deep sets (i.e., network for both Phi and Rho)
class InvestmentEuler(pl.LightningModule):
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
        # some general configuration
        verbose: bool,
        hpo_objective_name: str,
        print_metrics: bool,
        save_metrics: bool,
        save_test_results: bool,
        # parameters for method
        omega_quadrature_nodes: int,
        normalize_shock_vector: bool,
        train_trajectories: int,
        val_trajectories: int,
        test_trajectories: int,
        reset_trajectories_frequency: int,
        batch_size: int,
        shuffle_training: bool,
        T: int,
        X_0_loc: float,
        X_0_scale: float,
        # settings for deep learning approximation
        rho: torch.nn.Module,
        phi: torch.nn.Module,
    ):
        super().__init__()
        self.rho = rho
        self.phi = phi

        self.save_hyperparameters(ignore=["rho", "phi"])

        # Solves the LQ problem to find the comparison for the baseline
        # used for comparison as well as simulation of datapoints
        self.H_0, self.H_1 = self.investment_equilibrium_LQ(1)  # 1 firm is enough for

    # Calculates the LQ solution imposing symmetry by hand in the optimization process
    # Utility for direct comparison when a LQ solution is exact
    def investment_equilibrium_LQ(self, N):
        sigma, eta, alpha_0, alpha_1, delta, beta, gamma = (
            self.hparams.sigma,
            self.hparams.eta,
            self.hparams.alpha_0,
            self.hparams.alpha_1,
            self.hparams.delta,
            self.hparams.beta,
            self.hparams.gamma,
        )
        H_iv = [80.0, -0.2, 0.0]

        # Equation (22)
        B = np.zeros([N + 2, 1])
        B[1] = 1.0

        # Equation (23)
        C_1 = np.zeros([N + 1, N + 1])
        C_2 = np.zeros([1, N + 1])
        C_1[np.diag_indices(N + 1)] = sigma
        C_1[:, 0] = eta
        C_1[0, 1] = sigma
        C = np.concatenate((C_2, C_1))

        # Equation (24)
        R = np.zeros([N + 2, N + 2])
        R[1, :] = alpha_1 / (2 * N)
        R[:, 1] = alpha_1 / (2 * N)
        R[1, 1] = 0.0
        R[0, 1] = -alpha_0 / 2
        R[1, 0] = -alpha_0 / 2

        Q = gamma / 2

        # calculating A_hat
        def F_root(H):
            # Equation (30)
            H_0, H_1, H_2 = H  # H_2 not used

            # Equation (21)
            A = (H_1 / N) * np.ones([N + 2, N + 2])
            A[np.diag_indices(N + 2)] = 1.0 - delta + H_1 / N
            A[:, 0] = H_0
            A[:, 1] = 0.0
            A[0, :] = 0.0
            A[1, :] = 0.0
            A[0, 0] = 1.0
            A[1, 1] = 1.0 - delta

            lq = quantecon.LQ(Q, R, A, B, C, beta=beta)
            P, F, d = lq.stationary_values()
            return np.array([F[0][0], F[0][1], F[0][2]]) - np.array([-H[0], 0.0, -H[1] / N])

        H_opt = optimize.root(F_root, H_iv, method="lm", options={"xtol": 1.49012e-8})
        if not (H_opt.success):
            sys.exit("H optimization failed to converge.")

        H_hat = H_opt.x
        if self.hparams.verbose:
            print(f"LQ optima are: {H_hat}")
        return H_hat[0], H_hat[1]

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

    # model residuals given a set of states
    def model_residuals(self, X):
        u_X = self(X)

        # equation (12) and (13)
        X_primes = torch.stack(
            [
                u_X
                + (1 - self.hparams.delta) * X
                + self.hparams.sigma * self.expectation_shock_vector
                + self.hparams.eta * node
                for node in self.quadrature_nodes
            ]
        ).type_as(X)

        # p(X') calculation
        p_primes = self.hparams.alpha_0 - self.hparams.alpha_1 * X_primes.pow(self.hparams.nu).mean(
            2
        )

        # Expectation using quadrature over aggregate shock
        Ep = (p_primes.T @ self.quadrature_weights).type_as(X).reshape(-1, 1)

        Eu = (
            (
                torch.stack(tuple(self(X_primes[i]) for i in range(len(self.quadrature_nodes))))
                .squeeze(2)
                .T
                @ self.quadrature_weights
            )
            .type_as(X)
            .reshape(-1, 1)
        )

        # Euler equation itself
        residuals = self.hparams.gamma * u_X - self.hparams.beta * (
            Ep + self.hparams.gamma * (1 - self.hparams.delta) * Eu
        )  # equation (14)
        return residuals

    def training_step(self, X, batch_idx):
        residuals = self.model_residuals(X)

        loss = (residuals**2).sum() / len(residuals)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, X, batch_idx):
        residuals = self.model_residuals(X)

        loss = (residuals**2).sum() / len(residuals)

        self.log("val_loss", loss, prog_bar=True)

        # calculate policy error relative to analytic if linear
        if self.hparams.nu == 1:
            u_ref = self.linear_policy(X)
            u_rel_error = torch.mean(torch.abs(self(X) - u_ref) / torch.abs(u_ref))
            self.log("val_u_rel_error", u_rel_error, prog_bar=True)
            u_abs_error = torch.mean(torch.abs(self(X) - u_ref))
            self.log("val_u_abs_error", u_abs_error, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Test data includes trajectory number, time, etc.

        X = batch["X"]
        residuals = self.model_residuals(X)
        loss = (residuals**2).sum() / len(residuals)

        self.log("test_loss", loss, prog_bar=True)

        # Additional logging results
        if self.hparams.nu == 1:
            u_linear = self.linear_policy(X)
            u_X = self(X)
            u_rel_error = torch.abs(u_X - u_linear) / torch.abs(u_linear)
            u_abs_error = torch.abs(u_X - u_linear)

            self.test_results = pd.concat(
                [
                    self.test_results,
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
            self.test_results = pd.concat(
                [
                    self.test_results,
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

    # Data and simulation calculations
    # By default it uses the internal forward function, but can be overridden
    def simulate(self, num_trajectories, f = None, w=None, omega=None):
        # Simulates random numbers if not provided.
        if f is None:
            f = self.forward
        if w is None:
            w = torch.randn(
                num_trajectories,
                self.hparams.T,
                self.hparams.N,
                device=self.device,
                dtype=self.dtype,
            )
        if omega is None:
            omega = torch.randn(
                num_trajectories,
                self.hparams.T,
                1,
                device=self.device,
                dtype=self.dtype,
            )
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
                # Simulate using passed in "f",  which could be linear self.forward.
                f(data[:, t, :])  # num_ensembles by N
                + (1 - self.hparams.delta) * data[:, t, :]
                + self.hparams.sigma * w[:, t, :]
                + self.hparams.eta * omega[:, t]
            )
        return torch.cat(data.unbind(0))

    # Simulates all of the data using the state space model
    # At this point, the code is running local to the GPU/etc.
    def setup(self, stage):
        # quadrature for use within the expectation calculations
        nodes, weights = quantecon.quad.qnwnorm(self.hparams.omega_quadrature_nodes)
        self.quadrature_nodes = torch.tensor(nodes, dtype=self.dtype, device=self.device)
        self.quadrature_weights = torch.tensor(weights, dtype=self.dtype, device=self.device)

        # Monte Carlo draw for the expectations, possibly normalizing it
        vec = torch.randn(1, self.hparams.N, device=self.device, dtype=self.dtype)
        self.expectation_shock_vector = (
            (vec - vec.mean()) / vec.std() if self.hparams.normalize_shock_vector else vec
        )

        # Draw initial condition for the X_0 to simulate
        self.X_0_dist = torch.distributions.normal.Normal(  # not a tensor
            self.hparams.X_0_loc, self.hparams.X_0_scale
        )
        self.X_0 = torch.abs(self.X_0_dist.sample((self.hparams.N,)))

        if stage == "fit" or stage is None:
            # Simulate fixing the shock sequence
            self.train_data = self.simulate(self.hparams.train_trajectories, self.linear_policy)
            self.val_data = self.simulate(self.hparams.val_trajectories, self.linear_policy)

        if stage == "test" or stage is None:
            test_trajectories = self.hparams.test_trajectories
            # Note that this simulates with the built-in forward function itself, not the linear
            self.test_data = self.simulate(test_trajectories).reshape(
                [test_trajectories, self.hparams.T + 1, self.hparams.N]
            )

            # metadata zipping
            zipped = [
                {"ensemble": n, "t": t, "X": self.test_data[n, t, :]}
                for n in range(test_trajectories)
                for t in range(self.hparams.T + 1)
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

    # Reset simulation of training and validation data
    def training_epoch_end(self, outputs):
        # generates trajectories with current policy, regardless of nu
        if self.hparams.reset_trajectories_frequency > 0 and (self.current_epoch > 0) and (self.current_epoch % self.hparams.reset_trajectories_frequency == 0):
            self.train_data = self.simulate(self.hparams.train_trajectories)
            self.val_data = self.simulate(self.hparams.val_trajectories)


def log_and_save(trainer, model, train_time):
    save_path = trainer.log_dir

    # Setup to be wanddb centric for summary statistics/etc.
    if type(trainer.logger) is WandbLogger:
        # The calculated runtime with pytorch lightning + wandb has many fixed costs which throw off performance comparisons
        trainer.logger.experiment.log({"train_time": train_time})

        # Count and log the number of parameters with are trained in the neural network
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainer.logger.experiment.log({"trainable_parameters": trainable_parameters})

        # Set objective for hyperparameter optimization.  Only log if successful (i.e, val_loss < stopping_threshold)
        if hasattr(cli.trainer, "early_stopping_callback"):
            hpo_objective_value = dict(cli.trainer.logger.experiment.summary)[
                model.hparams.hpo_objective_name
            ]
            if (
                dict(cli.trainer.logger.experiment.summary)["val_loss"]
                < cli.trainer.early_stopping_callback.stopping_threshold
            ):
                trainer.logger.experiment.log({"hpo_objective": hpo_objective_value})
            else:
                trainer.logger.experiment.log({"hpo_objective": math.nan})

        # save the summary statistics in a file
        if model.hparams.save_metrics and save_path is not None:
            metrics_path = Path(save_path) / "metrics.yaml"
            with open(metrics_path, "w") as fp:
                yaml.dump(dict(cli.trainer.logger.experiment.summary), fp)

        if model.hparams.print_metrics:
            print(dict(cli.trainer.logger.experiment.summary))

        # Store the test_results field from model if it exists
        if hasattr(model, "test_results"):
            trainer.logger.log_text(
                key="test_results", dataframe=trainer.model.test_results
            )  # Saves on wandb for querying later
            if model.hparams.save_test_results and save_path is not None:
                model.test_results.to_csv(Path(save_path) / "test_results.csv", index=False)

    else:
        # otherwise just conditionally save test_results
        if (
            model.hparams.save_test_results
            and save_path is not None
            and hasattr(model, "test_results")
        ):
            model.test_results.to_csv(Path(save_path) / "test_results.csv", index=False)


if __name__ == "__main__":
    cli = LightningCLI(
        InvestmentEuler,
        seed_everything_default=123,
        run=False,
        save_config_callback=None,  # turn this on to save the full config file rather than just having it uploaded
        parser_kwargs={"default_config_files": ["investment_euler_defaults.yaml"]},
        save_config_kwargs={"save_config_overwrite": True},
    )

    # Fit the model.  Separating training time for plotting
    start = timeit.default_timer()
    cli.trainer.fit(cli.model)
    train_time = timeit.default_timer() - start

    # Check test data
    cli.trainer.test(cli.model)

    # Add additional calculations such as HPO objective to the log and save files
    log_and_save(cli.trainer, cli.model, train_time)
