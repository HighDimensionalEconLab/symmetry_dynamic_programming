import pandas as pd
import torch
import pytorch_lightning as pl
import yaml
import math
import numpy as np
import wandb
import timeit
import econ_layers
from torch.utils.data import DataLoader
from econ_layers.utilities import dict_to_cpu
from pytorch_lightning.cli import LightningCLI
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger


class GeneralizedMean(pl.LightningModule):
    def __init__(
        self,
        a_max: float,
        N: int,
        p: float,
        # some general configuration
        verbose: bool,
        hpo_objective_name: str,
        print_metrics: bool,
        save_metrics: bool,
        save_test_results: bool,
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

    # Used for evaluating u(X) given the current network
    def forward(self, X):
        return self.ml_model(X) # deep sets/etc.

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
            a_i = np.random.uniform(0, self.hparams.a_max)
            x_generator = torch.distributions.Uniform(a_i, 1 + a_i)
            X = x_generator.sample([self.hparams.N])
            y = X.pow(self.hparams.p).mean().pow(1 / self.hparams.p)  # generalized mean
            simulated_data.append((X, y.unsqueeze(0)))
        return simulated_data

    # At this point, the code is running local to the GPU/etc.
    def setup(self, stage):
        self.train_data = self.simulate_data(self.hparams.num_train_points)
        self.val_data = self.simulate_data(self.hparams.num_val_points)
        self.test_data = self.simulate_data(self.hparams.num_test_points)
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
    if model.hparams.save_test_results and trainer.log_dir is not None:
        model.test_results.to_csv(Path(trainer.log_dir) / "test_results.csv", index=False)
    if type(trainer.logger) is WandbLogger:
        # The calculated runtime with pytorch lightning + wandb has many fixed costs which throw off performance comparisons
        trainer.logger.experiment.log({"train_time": train_time})

        # Count and log the number of parameters with are trained in the neural network
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainer.logger.experiment.log({"trainable_parameters": trainable_parameters})

        # Set objective for hyperparameter optimization.  Only log if successful (e.g., val_loss < stopping_threshold)
        if hasattr(cli.trainer, "early_stopping_callback"):
            hpo_objective_value = dict(cli.trainer.logger.experiment.summary)[
                model.hparams.hpo_objective_name
            ]
            if (
                dict(cli.trainer.logger.experiment.summary)[cli.trainer.early_stopping_callback.monitor
] # e.g., `val_loss`
                < cli.trainer.early_stopping_callback.stopping_threshold
            ):
                trainer.logger.experiment.log({"hpo_objective": hpo_objective_value})
            else:
                trainer.logger.experiment.log({"hpo_objective": math.nan})

        # save the summary statistics in a file
        if model.hparams.save_metrics and trainer.log_dir is not None:
            metrics_path = Path(trainer.log_dir) / "metrics.yaml"
            with open(metrics_path, "w") as fp:
                yaml.dump(dict(cli.trainer.logger.experiment.summary), fp)

        if model.hparams.print_metrics:
            print(dict(cli.trainer.logger.experiment.summary))

        # Store the test_results field from model if it exists
        if hasattr(model, "test_results"):
            trainer.logger.log_text(
                key="test_results", dataframe=trainer.model.test_results
            )  # Saves on wandb for querying later


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
    cli.trainer.test(cli.model)

    # Add additional calculations such as HPO objective to the log and save files
    log_and_save(cli.trainer, cli.model, train_time)
