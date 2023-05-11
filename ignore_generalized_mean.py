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

class GeneralizedMean(pl.LightningModule):
    def __init__(
        self,
        a_min: float,
        a_max: float,
        X_distribution: str,
        std: float,
        N: int,
        p: float,
        # some general configuration
        verbose: bool,
        hpo_objective_name: str,
        always_log_hpo_objective: bool,
        print_metrics: bool,
        save_metrics: bool,
        save_test_results: bool,
        test_loss_success_threshold: float,
       
        ##do we need seed

        # parameters for method
        num_train_points: int,
        num_val_points: int,
        num_test_points: int,
        batch_size: int,
        shuffle_training: bool,
        # settings for deep learning approximation
        ml_model: torch.nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ml_model"])  # access with self.hparams.alpha, etc.
        self.ml_model = ml_model

# Used for evaluating the model
    def forward(self, X):
        return self.ml_model(X)  # deep sets/etc.

    def training_step(self, batch, batch_idx):
        x, y = batch
        residuals = y - self(x)
        loss = (residuals**2).sum() / len(residuals)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        residuals = y - self(x)
        loss = (residuals**2).sum() / len(residuals)

        rel_error = torch.mean(torch.abs(residuals) / torch.abs(y))
        abs_error = torch.mean(torch.abs(residuals))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rel_error", rel_error, prog_bar=True)
        self.log("val_abs_error", abs_error, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_f = batch
        y = self(x)
        residuals = y_f - y
        loss = (residuals**2).sum() / len(residuals)
        rel_error = torch.abs(y_f - y) / torch.abs(y_f)
        abs_error = torch.abs(y_f - y)

        self.test_results = pd.concat(
            [
                self.test_results,
                pd.DataFrame(
                    dict_to_cpu(
                        {
                            "x_norm": x.norm(dim=1),  # x is too large to store
                            "f_x": y_f,
                            "f_hat_x": y,
                            "rel_error": rel_error,
                            "abs_error": abs_error,
                        }
                    )
                ),
            ]
        )
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_rel_error", rel_error.mean(), prog_bar=True)
        self.log("test_abs_error", abs_error.mean(), prog_bar=True)

    # simulate DGP
    def simulate_data(self, num_points):
        simulated_data = []
        for i in range(0, num_points):
            a_i = np.random.uniform(self.hparams.a_min, self.hparams.a_max)
            if self.hparams.X_distribution=="normal":
                X = torch.normal(a_i, self.hparams.std, size=(self.hparams.N,))
            elif self.hparams.X_distribution=="uniform":
                d = self.hparams.std * math.sqrt(3) # ensures std is correct
                X = torch.rand(self.hparams.N) * 2 * d + a_i - d # uniform in [a_i - d, a_i + d]
            else:
                raise ValueError("Distribution not supported")
            y = X.pow(self.hparams.p).mean().pow(1 / self.hparams.p)  # generalized mean
            simulated_data.append((X, y.unsqueeze(0)))
        return simulated_data

    # At this point, the code is running local to the GPU/etc.
    def setup(self, stage):
        self.train_data = self.simulate_data(self.hparams.num_train_points)
        self.val_data = self.simulate_data(self.hparams.num_val_points)
        self.test_data = self.simulate_data(self.hparams.num_test_points)
        self.test_data = self.simulate(self.hparams.num_test_points).reshape([test_trajectories, self.hparams.T + 1, self.hparams.N])
        self.test_data = [
            {   "X": self.test_data[n, t, :],
                "y_t": self.test_data[n, t, :]????
            }
            for n in range(test_trajectories)
            for t in range(self.hparams.T + 1)
        ] 
        self.test_results = pd.DataFrame()

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

            return False  # otherwise a valid, non-infinite number

        # If early stopping, evaluate success
        early_stopping_check_failed = math.nan
        early_stopping_monitor = ""
        early_stopping_threshold = math.nan
        for callback in trainer.callbacks:
            if type(callback) == pl.callbacks.early_stopping.EarlyStopping:
                early_stopping_monitor = callback.monitor
                early_stopping_threshold = callback.stopping_threshold
                early_stopping_check_failed = not_number_type(
                    cli.trainer.logger.experiment.summary[callback.monitor]
                ) or (
                    cli.trainer.logger.experiment.summary[callback.monitor]
                    > callback.stopping_threshold
                )
                break


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
            and test_loss_check_failed in [False, math.nan]
        ):
            retcode = 0
            convergence_description = "Success"
        elif early_stopping_check_failed == True:
            retcode = -1
            convergence_description = "Early stopping failure"
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
        trainer.logger.experiment.log({"test_loss_check_failed": test_loss_check_failed})
        trainer.logger.experiment.log({"trainable_parameters": trainable_parameters})
        trainer.logger.experiment.log({"retcode": retcode})
        trainer.logger.experiment.log({"convergence_description": convergence_description})

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
        GeneralizedMean,
        seed_everything_default=155,
        run=False,
        save_config_callback=None,  # turn this on to save the full config file rather than just having it uploaded
        parser_kwargs={"default_config_files": ["generalized_mean_defaults_E.yaml"]},
        save_config_kwargs={"save_config_overwrite": True},
    )
    # Fit the model.  Separating training time for plotting, and evaluate generalization
    start = timeit.default_timer()
    cli.trainer.fit(cli.model)
    train_time = timeit.default_timer() - start
    cli.trainer.test(cli.model)

    # Add additional calculations such as HPO objective to the log and save files
    log_and_save(cli.trainer, cli.model, train_time)
