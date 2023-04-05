import pandas as pd
import torch
import pytorch_lightning as pl
import yaml
import math
import numpy as np
import scipy
import wandb
import timeit
import quantecon
import econ_layers
import scipy.optimize
from torch.utils.data import DataLoader
from econ_layers.utilities import dict_to_cpu
from pytorch_lightning.cli import LightningCLI
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger


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
        always_log_hpo_objective: bool,
        print_metrics: bool,
        save_metrics: bool,
        save_test_results: bool,
        test_seed: int,
        check_transversality: bool,
        test_loss_success_threshold: float,
        transversality_X_mean_min: float,
        transversality_X_mean_max: float,
        transversality_u_rel_error: float,
        # parameters for method
        omega_quadrature_nodes: int,
        normalize_shock_vector: bool,
        train_trajectories: int,
        val_trajectories: int,
        test_trajectories: int,
        train_subsample_trajectories: int,
        reset_trajectories_frequency: int,
        batch_size: int,
        shuffle_training: bool,
        T: int,
        X_0_loc: float,
        X_0_scale: float,
        # settings for deep learning approximation
        ml_model: torch.nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ml_model"])  # access with self.hparams.alpha, etc.
        self.ml_model = ml_model
        # Solves the LQ problem to find the comparison for the nu=1 case and generating simulations
        self.H_0, self.H_1 = self.investment_equilibrium_LQ()  # 1 firm is enough for

    # Calculates the LQ solution imposing symmetry by hand in the optimization process
    def investment_equilibrium_LQ(self):
        B = np.array([[0.0], [1.0], [0.0]])
        C = np.array(
            [
                [0.0, 0.0],
                [self.hparams.eta, self.hparams.sigma],
                [self.hparams.eta, self.hparams.sigma],
            ]
        )
        R = np.array(
            [
                [0.0, -self.hparams.alpha_0 / 2, 0.0],
                [-self.hparams.alpha_0 / 2, 0.0, self.hparams.alpha_1 / 2],
                [0.0, self.hparams.alpha_1 / 2, 0.0],
            ]
        )  # Equation (24)
        Q = self.hparams.gamma / 2

        # calculating A_hat
        def F_root(H):
            A = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0 - self.hparams.delta, 0.0],
                    [H[0], 0.0, 1.0 - self.hparams.delta + H[1]],
                ]
            )  # Equation (21)
            lq = quantecon.LQ(Q, R, A, B, C, beta=self.hparams.beta)
            P, F, d = lq.stationary_values()
            return np.array([F[0][0], F[0][1], F[0][2]]) - np.array([-H[0], 0.0, -H[1]])

        H_opt = scipy.optimize.root(F_root, [80.0, -0.2], method="lm", options={"xtol": 1.49012e-8})
        if not (H_opt.success):
            sys.exit("H optimization failed to converge.")
        return H_opt.x[0], H_opt.x[1]

    # Used for evaluating u(X) given the current network
    def forward(self, X):
        return self.ml_model(X)  # deep sets/etc.

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
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, X, batch_idx):
        residuals = self.model_residuals(X)
        loss = (residuals**2).sum() / len(residuals)
        self.log("val_loss", loss, prog_bar=True)

        # calculate policy error relative to analytic if linear
        if self.hparams.nu == 1:
            u_ref = self.H_0 + self.H_1 * X.mean(1, keepdim=True)  # closed form if linear
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
            u_linear = self.H_0 + self.H_1 * X.mean(1, keepdim=True)  # closed form if linear
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
                                "u_rel_error": u_rel_error,
                                "u_abs_error": u_abs_error,
                                "X_min": batch["X_min"],
                                "X_max": batch["X_max"],
                                "X_mean": batch["X_mean"],
                                "X_std": batch["X_std"],
                            }
                        )
                    ),
                ]
            )
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
                                "X_min": batch["X_min"],
                                "X_max": batch["X_max"],
                                "X_mean": batch["X_mean"],
                                "X_std": batch["X_std"],
                            }
                        )
                    ),
                ]
            )

    # Data and simulation calculations.
    def simulate(self, X_0, num_trajectories, f=None, w=None, omega=None):
        # Simulates random numbers if not provided.
        if f is None:
            f = self.forward  # use the self.forward(..) by default
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

        data[:, 0, :] = X_0
        for t in range(0, self.hparams.T):
            data[:, t + 1, :] = (
                # Simulate using passed in "f",  which could be linear self.forward.
                f(data[:, t, :])  # num_ensembles by N
                + (1 - self.hparams.delta) * data[:, t, :]
                + self.hparams.sigma * w[:, t, :]
                + self.hparams.eta * omega[:, t]
            )
        return torch.cat(data.unbind(0))

    # Setup data/etc.  Supposed to be in setup instead of the __init__
    def setup(self, stage):
        if stage == "fit" or stage is None:
            # quadrature for use within the expectation calculations
            nodes, weights = quantecon.quad.qnwnorm(self.hparams.omega_quadrature_nodes)
            nodes = torch.tensor(nodes, device=self.device, dtype=self.dtype)
            weights = torch.tensor(weights, device=self.device, dtype=self.dtype)

            # Monte Carlo draw for the expectations, possibly normalizing it
            vec = torch.randn(1, self.hparams.N, device=self.device, dtype=self.dtype)
            expectation_shock_vector = (
                (vec - vec.mean()) / vec.std() if self.hparams.normalize_shock_vector else vec
            )

            # Draw initial condition for the X_0 to simulate
            X_0_dist = torch.distributions.normal.Normal(  # not a tensor
                self.hparams.X_0_loc, self.hparams.X_0_scale
            )
            X_0 = torch.abs(X_0_dist.sample((self.hparams.N,))).type_as(expectation_shock_vector)

            # Use a linear policy for initial simulation: h_0 + h_1 mean(X). h_0>0, h_1<0 guarantees stationarity and positivity. |h_0/h_1|<1 guarantees prices p(X)>0 in the sample
            def initial_trajectory_policy(X):
                return self.H_0 + self.H_1 * X.mean(1, keepdim=True)

            train_data = self.simulate(X_0, 
                self.hparams.train_trajectories, initial_trajectory_policy
            ).type_as(expectation_shock_vector)
            if self.hparams.train_subsample_trajectories > 0:
                sample_idx = np.random.randint(len(train_data), size=self.hparams.train_subsample_trajectories)
                train_data = train_data[sample_idx]
            val_data = self.simulate(X_0, self.hparams.val_trajectories, initial_trajectory_policy).type_as(expectation_shock_vector)

            # Store buffers for optimization.  Replaces assignment toensure it is transfered to GPU/etc. properly
            self.register_buffer("quadrature_nodes", nodes) # i.e., instead of self.quadrature_nodes = nodes
            self.register_buffer("quadrature_weights", weights)
            self.register_buffer("expectation_shock_vector", expectation_shock_vector)
            self.register_buffer("X_0", X_0)
            self.register_buffer("train_data", train_data)
            self.register_buffer("val_data", val_data)

        if stage == "test":

            # Initial conditions and vectors for shocks are identical to those in the first stages
            
            if self.hparams.test_seed > 0:
                pl.seed_everything(
                    self.hparams.test_seed
                )  # set seed separate for generating test data

            test_trajectories = self.hparams.test_trajectories
            # Note that this simulates with the built-in forward function itself, not the linear
            self.test_data = self.simulate(self.X_0, test_trajectories).reshape(
                [test_trajectories, self.hparams.T + 1, self.hparams.N]
            )
            self.test_data = [
                {
                    "ensemble": n,
                    "t": t,
                    "X": self.test_data[n, t, :],
                    "X_min": self.test_data[n, t, :].min(),
                    "X_max": self.test_data[n, t, :].max(),
                    "X_mean": self.test_data[n, t, :].mean(),
                    "X_std": self.test_data[n, t, :].std(),
                }
                for n in range(test_trajectories)
                for t in range(self.hparams.T + 1)
            ]  # includes ensemble information for analysis
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
    def on_train_epoch_end(self):
        # generates trajectories with current policy, regardless of nu
        if (
            self.hparams.reset_trajectories_frequency > 0
            and (self.current_epoch > 0)
            and (self.current_epoch % self.hparams.reset_trajectories_frequency == 0)
        ):
            self.train_data = self.simulate(self.hparams.train_trajectories)
            self.val_data = self.simulate(self.hparams.val_trajectories)


def log_and_save(trainer, model, train_time):
    if type(trainer.logger) is WandbLogger:
        # Valid numeric types
        def not_number_type(value):
            if value is None:
                return True

            if not isinstance(value, (int, float)):
                return True

            if math.isnan(value) or math.isinf(value):
                return True

            return False # otherwise a valid, non-infinite number


        # If early stopping, evaluate success
        early_stopping_check_failed = math.nan
        early_stopping_monitor = ""
        early_stopping_threshold = math.nan
        for callback in trainer.callbacks:
            if type(callback) == pl.callbacks.early_stopping.EarlyStopping:
                early_stopping_monitor = callback.monitor
                early_stopping_threshold = callback.stopping_threshold
                early_stopping_check_failed = not_number_type(cli.trainer.logger.experiment.summary[callback.monitor]) or (
                    cli.trainer.logger.experiment.summary[callback.monitor]
                    > callback.stopping_threshold
                )
                break

        # Check transversality
        X_T_mean = trainer.model.test_results.loc[
            trainer.model.test_results["t"] == model.hparams.T
        ].X_mean.mean()
        
        # if nu = 1 it is more robust to check the u_rel_error, otherwise assume T is large enough that divergence would occur for X_T
        if not model.hparams.check_transversality:
            transversality_check_failed = math.nan
        elif (model.hparams.nu == 1) and (
            not_number_type(cli.trainer.logger.experiment.summary["val_u_rel_error"])
            or (
                cli.trainer.logger.experiment.summary["val_u_rel_error"]  # known at validation time
                > trainer.model.hparams.transversality_u_rel_error
            )
        ):
            transversality_check_failed = True
        elif (model.hparams.nu != 1) and (
            not_number_type(X_T_mean)
            or (X_T_mean < model.hparams.transversality_X_mean_min)
            or (X_T_mean > model.hparams.transversality_X_mean_max)
        ):
            transversality_check_failed = True
        else:
            transversality_check_failed = False

        # Check test loss
        if model.hparams.test_loss_success_threshold == 0:
            test_loss_check_failed = math.nan
        elif not_number_type(cli.trainer.logger.experiment.summary["test_loss"]) or (
            cli.trainer.logger.experiment.summary["test_loss"]
            > model.hparams.test_loss_success_threshold
        ):
            test_loss_check_failed = True
        else:
            test_loss_check_failed = False

        # Determine convergence results
        if (
            early_stopping_check_failed in [False, math.nan]
            and transversality_check_failed in [False, math.nan]
            and test_loss_check_failed in [False, math.nan]
        ):
            retcode = 0
            convergence_description = "Success"
        elif early_stopping_check_failed == True:
            retcode = -1
            convergence_description = "Early stopping failure"
        elif transversality_check_failed == True:
            retcode = -2
            convergence_description = "Transversality check failure"  # possible due to finding wrote root but could also be other issues which manifest as a transversality violation
        elif test_loss_check_failed == True:
            retcode = -3
            convergence_description = "Test loss failure due to possible overfitting."  # if nu != 1 but T was set low, this might also be due to transversality failures
        else:
            retcode = -100
            convergence_description = " Unknown failure"

        # Log all calculated results
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainer.logger.experiment.log({"train_time": train_time})
        trainer.logger.experiment.log({"early_stopping_monitor": early_stopping_monitor})
        trainer.logger.experiment.log({"early_stopping_threshold": early_stopping_threshold})
        trainer.logger.experiment.log({"early_stopping_check_failed": early_stopping_check_failed})
        trainer.logger.experiment.log({"transversality_check_failed": transversality_check_failed})
        trainer.logger.experiment.log({"test_loss_check_failed": test_loss_check_failed})
        trainer.logger.experiment.log({"trainable_parameters": trainable_parameters})
        trainer.logger.experiment.log({"retcode": retcode})
        trainer.logger.experiment.log({"convergence_description": convergence_description})
        trainer.logger.experiment.log({"X_T_mean": X_T_mean})        

        # Set objective for hyperparameter optimization
        # Objective value given in the settings, or empty
        if model.hparams.hpo_objective_name is not None:
            hpo_objective_value = dict(cli.trainer.logger.experiment.summary)[
                model.hparams.hpo_objective_name
            ]
        else:
            hpo_objective_value = math.nan
                    
        if model.hparams.always_log_hpo_objective or retcode >= 0:
            trainer.logger.experiment.log({"hpo_objective": hpo_objective_value})
        else:
            trainer.logger.experiment.log({"hpo_objective": math.nan})

        # Save test results
        trainer.logger.log_text(
            key="test_results", dataframe=trainer.model.test_results
        )  # Saves on wandb for querying later

        # save the summary statistics in a file
        if model.hparams.save_metrics and trainer.log_dir is not None:
            metrics_path = Path(trainer.log_dir) / "metrics.yaml"
            with open(metrics_path, "w") as fp:
                yaml.dump(dict(cli.trainer.logger.experiment.summary), fp)

        if model.hparams.print_metrics:
            print(dict(cli.trainer.logger.experiment.summary))
        return
    else:  # almost no features enabled for other loggers. Could refactor later
        if model.hparams.save_test_results and trainer.log_dir is not None:
            model.test_results.to_csv(Path(trainer.log_dir) / "test_results.csv", index=False)


if __name__ == "__main__":
    cli = LightningCLI(
        InvestmentEuler,
        seed_everything_default=123,
        run=False,
        save_config_callback=None,  # turn this on to save the full config file rather than just having it uploaded
        parser_kwargs={"default_config_files": ["investment_euler_defaults.yaml"]},
        save_config_kwargs={"save_config_overwrite": True},
    )
    # Fit the model.  Separating training time for plotting, and evaluate generalization
    start = timeit.default_timer()
    cli.trainer.fit(cli.model)
    train_time = timeit.default_timer() - start
    cli.trainer.test(cli.model)

    # Add additional calculations such as HPO objective to the log and save files
    log_and_save(cli.trainer, cli.model, train_time)


