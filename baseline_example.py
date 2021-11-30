import pandas as pd
import torch
import pytorch_lightning as pl
import yaml

from torch import nn
from jsonargparse import lazy_instance
from torch.utils.data import DataLoader
from econ_layers.utilities import dict_to_cpu
from pytorch_lightning.utilities.cli import LightningCLI
from pathlib import Path
from copy import deepcopy

# Local packages
from .linear_policy_LQ import investment_equilibrium_LQ


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
        zeta: float,
        # parameters for method
        verbose: bool,
        W_quadrature_nodes: int,
        normalize_shock_vector: bool,
        train_trajectories: int,
        val_trajectories: int,
        test_trajectories: int,
        batch_size: int,
        T: int,
        X_0_loc: float,
        X_0_scale: float,
        # settings for deep learning approximation
        phi_activator: lazy_instance(nn.ReLU),
        phi_layers: int,
        phi_dim: int,
        phi_bias: bool,
        rho_activator: lazy_instance(nn.ReLU),
        rho_layers: int,
        rho_dim: int,
        rho_bias: bool,
        L: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Networks for the Function Representation
        self.rho = nn.Sequential(
            nn.Linear(L, rho_dim, bias=rho_bias),
            deepcopy(rho_activator)
            # Add in rho_layers - 1
            * [
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
        self.H_0, self.H_1 = investment_equilibrium_LQ(1, self.hparams)
        # The "simulation_policy" function starts by using the linear_policy
        # to begin the simulation of X_t grid points.  Swaps out later
        self.simulation_policy = self.linear_policy

    # Used for evaluating the current approximation
    def forward(self, X):
        num_batches, N = X.shape

        # Apply network with the representation and "mean" pooling
        phi_X = torch.stack(
            [
                torch.mean(self.phi(X[i, :].reshape([N, 1])), 0)
                for i in range(num_batches)
            ]
        )
        return self.rho(phi_X)

    # An analytic linear policy for simulation and comparison.  Uses LQ solution
    # Exact if \nu = 1.  Used for generating grid of data, not fitting itself.
    def linear_policy(self, X):
        return self.H_0 + self.H_1 * X.mean(1, keepdim=True)

    # model residuals given a set of states
    def model_residuals(self, x):
        # y = (
        #     torch.matmul(self.G, x.t()).unsqueeze(0).t()
        # )  # broadcasts and shapes to align for residual broadcasting
        # # difference equation: p(x) = y(x) + beta * p(x')
        # # here: p(x) = G x + beta * p(A x)
        # # x_t = torch.stack([torch.zeros(len(y), device = self.device, dtype = self.dtype), torch.ones(len(y), device = self.device, dtype = self.dtype), y])
        # x_tp1 = torch.matmul(self.A, x.t()).t()
        # y_tp1 = torch.matmul(self.G, x_tp1.t()).unsqueeze(0).t()

        # # x_p = torch.matmul(self.A, x.t()).t()  # broadcasts x_p = A x
        # residuals = (
        #     y + self.hparams.beta * self(y_tp1) - self(y)
        # )  # TODO: check broadcasting in debugger!
        return residuals

    # Utility function which makes the batch better for broadcasting.
    def reshape_batch(self, batch):
        x_t = batch
        return x_t

    def reshape_batch_test(self, batch):
        samples, t, y_t, p_f_t, x_t = batch
        samples = samples.reshape(len(batch[0]), 1)
        t = t.reshape(len(batch[0]), 1)
        y_t = y_t.reshape(len(batch[0]), 1)
        p_f_t = p_f_t.reshape(len(batch[0]), 1)

        return samples, t, y_t, p_f_t, x_t

    def training_step(self, batch, batch_idx):
        x = self.reshape_batch(batch)

        residuals = self.residuals(x)
        loss = (residuals ** 2).sum() / len(residuals)

        self.log("train_loss", loss)
        self.log("epoch", self.current_epoch, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.reshape_batch(batch)

        residuals = self.residuals(x)
        loss = (residuals ** 2).sum() / len(residuals)

        # bubble_mse = torch.nn.functional.mse_loss(self(x), p_f)

        self.log("val_loss", loss, prog_bar=True)
        # self.log("bubble_mse", bubble_mse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass
        # samples, t, y, p_f, x = self.reshape_batch_test(batch)
        # y_init = self.y_0[samples]

        # # get fundamentals
        # y_t = torch.matmul(self.G, x.t()).unsqueeze(0).t()
        # p = self(y_t)
        # p_bubble = p - p_f
        # p_error = p_bubble / p_f
        # bubble_mse = torch.nn.functional.mse_loss(p, p_f)
        # residuals = self.residuals(x)
        # loss = (residuals ** 2).sum() / len(residuals)
        # self.test_results = pd.concat(
        #     [
        #         self.test_results,
        #         pd.DataFrame(
        #             dict_to_cpu(
        #                 {
        #                     "sample": samples,
        #                     "t": t,  # just the t index for now
        #                     "y_0 ": y_init,
        #                     "y_t": y_t,
        #                     "residuals": residuals,
        #                     "p_t": p,
        #                     "p_f_t": p_f,
        #                     "p_bubble": p_bubble,
        #                     "p_error": p_error,
        #                 }
        #             )
        #         ),
        #     ]
        # )
        # self.log("bubble_mse", bubble_mse)
        # self.log("test_loss", loss, prog_bar=True)

    ## Data and simulation calculations

    # Simulates all of the data using the state space model
    def setup(self, stage):
        pass
        # if stage == "fit" or stage is None:
        #     y = torch.linspace(
        #         self.hparams.y_grid_min,
        #         self.hparams.y_grid_max,
        #         self.hparams.sim_grid_points,
        #     )

        #     self.simulated_data = []
        #     self.simulated_data = list()

        #     for i in range(0, self.hparams.sim_grid_points):
        #         # x_i = torch.stack([torch.zeros(1, device = self.device, dtype = self.dtype), torch.ones(1, device = self.device, dtype = self.dtype), y[i]*torch.ones(1, device = self.device, dtype = self.dtype)])
        #         x_i = torch.tensor([0.0, 1.0, y[i]])
        #         self.simulated_data.append(x_i)

        #     # y = torch.arange(min_value, max_value, step)
        #     # x = torch.stack([torch.zeros(len(y), device = self.device, dtype = self.dtype), torch.ones(len(y), device = self.device, dtype = self.dtype), y])

        #     # self.simulated_data = list(x)

        #     self.train_data = self.simulated_data
        #     self.val_data = self.train_data
        # elif stage == "test":
        #     self.simulated_data = []
        #     self.simulated_data = list()
        #     x_t = self.x_0
        #     n_trajectories = len(self.y_0)
        #     for t in torch.arange(0.0, self.hparams.max_T_test):
        #         y_t = self.G @ x_t
        #         p_f_t = self.H @ x_t
        #         # appends each trajectory separately
        #         for i in range(0, n_trajectories):
        #             data_i_t = (i, t, y_t[i], p_f_t[i], x_t[:, i])
        #             self.simulated_data.append(data_i_t)
        #         x_t = self.A @ x_t  # iterate forward
        #     self.test_data = self.simulated_data

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle_training,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
        )


def save_results(trainer, model, metrics_dict, print_metrics=True):
    metrics_path = Path(trainer.log_dir) / "metrics.yaml"
    if print_metrics:
        print(metrics_dict)

    with open(metrics_path, "w") as fp:
        yaml.dump(metrics_dict, fp)

    # Store the test_results field on model if it exists
    if hasattr(model, "test_results"):
        model.test_results.to_csv(
            Path(trainer.log_dir) / "test_results.csv", index=False
        )


def cli_main():
    # See config_defaults.yaml for defaults on the trainers/etc.)
    cli = LightningCLI(
        InvestmentEulerBaseline,
        seed_everything_default=123,
        save_config_overwrite=True,
    )

    # Fit the model
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
