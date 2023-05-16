import pandas as pd
import torch
import pytorch_lightning as pl
import yaml
import math
import numpy as np
import wandb
import timeit
import econ_layers
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.cli import LightningCLI
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import TensorDataset


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
        test_seed: int,
        train_data_seed: int,
        test_loss_success_threshold: float,
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
        y = y.unsqueeze(1)  # to enable broadcasting of self(x)
        loss = F.mse_loss(self(x), y, reduction="mean")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)  # to enable broadcasting of self(x)
        residuals = y - self(x)
        loss = F.mse_loss(self(x), y, reduction="mean")

        rel_error = torch.mean(torch.abs(residuals) / torch.abs(y))
        abs_error = torch.mean(torch.abs(residuals))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rel_error", rel_error, prog_bar=True)
        self.log("val_abs_error", abs_error, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_f = batch
        y_f = y_f.unsqueeze(1)  # to enable broadcasting of self(x)
        y = self(x)
        residuals = y_f - y
        loss = F.mse_loss(y_f, y, reduction="mean")
        rel_error = torch.abs(y_f - y) / torch.abs(y_f)
        abs_error = torch.abs(y_f - y)

        self.test_results = pd.concat(
            [
                self.test_results,
                pd.DataFrame(
                    {
                        "x_norm": x.norm(dim=1).cpu().numpy().tolist(),  # x is too large to store
                        "f_x": y_f.squeeze().cpu().numpy().tolist(),
                        "f_hat_x": y.squeeze().cpu().numpy().tolist(),
                        "rel_error": rel_error.squeeze().cpu().numpy().tolist(),
                        "abs_error": abs_error.squeeze().cpu().numpy().tolist(),
                    }
                ),
            ]
        )
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_rel_error", rel_error.mean(), prog_bar=True)
        self.log("test_abs_error", abs_error.mean(), prog_bar=True)

    # simulate DGP
    def simulate_data(self, num_points, generator=None):
        X = torch.empty(num_points, self.hparams.N, device=self.device, dtype=self.dtype)
        for i in range(0, num_points):
            a_i = torch.empty(1).uniform_(
                self.hparams.a_min, self.hparams.a_max, generator=generator
            )
            if self.hparams.X_distribution == "normal":
                X[i] = torch.normal(
                    a_i.squeeze(0),
                    self.hparams.std,
                    size=(self.hparams.N,),
                    device=self.device,
                    dtype=self.dtype,
                    generator=generator,
                )
            elif self.hparams.X_distribution == "uniform":
                d = self.hparams.std * math.sqrt(3)  # ensures std is correct
                X[i] = (
                    torch.rand(
                        self.hparams.N, device=self.device, dtype=self.dtype, generator=generator
                    )
                    * 2
                    * d
                    + a_i
                    - d
                )  # uniform in [a_i - d, a_i + d]
            else:
                raise ValueError("Distribution not supported")

        Y = (
            X.pow(self.hparams.p).mean(dim=1).pow(1 / self.hparams.p)
        )  # generalized mean  Doing mean over each row
        return X, Y

    def setup(self, stage):
        if stage == "fit" or stage is None:
            if self.hparams.train_data_seed > 0:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self.hparams.train_data_seed)
            else:
                generator = None  # otherwise use default RNG

            # self.train_data = self.old_simulate_data(self.hparams.num_train_points)
            X, Y = self.simulate_data(self.hparams.num_train_points, generator=generator)
            self.train_data = TensorDataset(X, Y)

            if self.hparams.num_val_points > 0:
                X, Y = self.simulate_data(self.hparams.num_val_points, generator=generator)
                self.val_data = TensorDataset(X, Y)
            else:
                self.val_data = []
        if stage == "test":
            if self.hparams.test_seed > 0:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self.hparams.test_seed)
            else:
                generator = None  # otherwise use default RNG

            X, Y = self.simulate_data(self.hparams.num_test_points, generator=generator)
            self.test_data = TensorDataset(X, Y)
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

def log_and_save(trainer, model, train_time, train_callback_metrics):
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
                early_stopping_value = train_callback_metrics[callback.monitor].cpu().numpy().tolist()
                early_stopping_threshold = callback.stopping_threshold
                early_stopping_check_failed = not_number_type(early_stopping_value
                ) or (early_stopping_value > callback.stopping_threshold)  # hardcoded to min for now.
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
            convergence_description = "Test loss failure due to possible overfitting"
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
        seed_everything_default=123,
        run=False,
        save_config_callback=None,  # turn this on to save the full config file rather than just having it uploaded
        parser_kwargs={"default_config_files": ["generalized_mean_defaults.yaml"]},
        save_config_kwargs={"save_config_overwrite": True},
    )
    # Fit the model.  Separating training time for plotting, and evaluate generalization
    start = timeit.default_timer()
    cli.trainer.fit(cli.model)
    train_time = timeit.default_timer() - start
    train_callback_metrics = cli.trainer.callback_metrics
    cli.trainer.test(cli.model)

    # Add additional calculations such as HPO objective to the log and save files
    log_and_save(cli.trainer, cli.model, train_time, train_callback_metrics)
