import argparse
import os
from argparse import ArgumentParser
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision

from models.modules import Predictor, Projector

video_model_names = sorted(
    name
    for name in torchvision.models.video.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision.models.video.__dict__[name])
)


def prefix_metrics_keys(
    metrics_dict: Dict[str, float], prefix: str
) -> Dict[str, float]:
    return {prefix + "." + k: v for k, v in metrics_dict.items()}


class Brave(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        num_features: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student = torchvision.models.video.__dict__[arch](num_classes=num_features)
        self.teacher = torchvision.models.video.__dict__[arch](num_classes=num_features)
        self.projector = Projector(num_features, output_dim, hidden_dim)
        self.predictor = Predictor(output_dim, output_dim, hidden_dim)

    def forward(self, batch):
        # Compute student embedding.
        video, labels = batch
        embeddings = self.student(video.float())
        return embeddings, labels

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def test_step(self, batch, batch_idx):
        return self(batch)

    def training_step(self, batch, batch_idx):
        # Get student and teacher videos.
        x_s, x_t = batch[0].float(), batch[1].float()

        # Compute student and teacher embeddings.
        f_s = self.student(x_s)
        f_t = self.teacher(x_t)

        # Compute projected embeddings.
        z_s = self.projector(f_s)
        z_t = self.projector(f_t)

        # Compute embedding predictions.
        h_s = self.predictor(z_s)
        h_t = self.predictor(z_t)

        # Regress student prediction on teacher projection.
        loss_s_to_t = F.mse_loss(F.normalize(h_s), F.normalize(z_t).detach())

        # Regress teacher prediction on student projection.
        loss_t_to_s = F.mse_loss(F.normalize(h_t), F.normalize(z_s).detach())

        # Aggregate student and teacher losses.
        loss = loss_s_to_t + loss_t_to_s

        # Log times.
        profiler_times = self.trainer.profiler.recorded_durations
        if len(profiler_times["run_training_batch"]) > 0:
            compute_time = np.array(profiler_times["run_training_batch"]).mean()
            data_time = np.array(profiler_times["get_train_batch"]).mean()
        else:
            compute_time = -1.0
            data_time = -1.0

        self.log("compute_time", compute_time, prog_bar=True)
        self.log("data_time", data_time, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        # Outputs is a list of dict output for exach optimizer from the training step.
        self.log(
            "loss",
            torch.mean(torch.stack(tuple(out["loss"] for out in outputs))),
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # Predictor parameters have a 10x learning rate.
        backbone_params, predictor_params = [], []
        for name, param in self.named_parameters():
            if "predictor" in name:
                predictor_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": self.hparams.learning_rate},
                {"params": predictor_params, "lr": self.hparams.learning_rate * 10},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.3, patience=3, verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser, hyperopt: bool = False):
        if not hyperopt:
            parser = ArgumentParser(parents=[parser], add_help=False)
            parser.add_argument("--learning_rate", default=0.01, type=float)

        else:
            parser.opt_list(
                "--learning_rate",
                default=1.0,
                # options=[10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003],
                options=[10.0, 3.0],
                type=float,
                tunable=True,
            )

        parser.add_argument(
            "--arch", default="r3d_18", choices=["mc3_18", "r2plus1d_18", "r3d_18"]
        )
        parser.add_argument("--num_features", default=2048, type=int)
        parser.add_argument("--hidden_dim", default=4096, type=int)
        parser.add_argument("--output_dim", default=128, type=int)

        return parser
