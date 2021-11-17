from argparse import ArgumentParser
from typing import Any, Dict

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


class Brave(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        num_features: int,
        output_dim: int,
        optimizer_config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student = torchvision.models.video.__dict__[arch](num_classes=num_features)
        self.teacher = torchvision.models.video.__dict__[arch](num_classes=num_features)
        self.projector = Projector(num_features, output_dim)
        self.predictor = Predictor(output_dim, output_dim)

    def forward(self, batch):
        # Get student and teacher batch.
        video, audio = batch
        x_s, x_t = video

        print(x_s.shape, x_t.shape)
        assert 1 == 0

        x_s = torch.transpose(x_s, 2, 1)
        x_t = torch.transpose(x_t, 2, 1)

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
        return loss

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"Loss/train": 0})

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        return loss

    def training_epoch_end(self, outputs):
        # Outputs is a list of dict output for exach optimizer from the training step.
        self.log(
            "Loss/train",
            torch.mean(torch.stack(tuple(out["loss"] for out in outputs))),
            on_epoch=True,
        )

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"Loss/val": 0})

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        return loss

    def val_epoch_end(self, outputs):
        # Outputs is a list of dict output for exach optimizer from the training step.
        self.log(
            "Loss/val",
            torch.mean(torch.stack(tuple(out["loss"] for out in outputs))),
            on_epoch=True,
        )

    def configure_optimizers(self):
        config = dict(self.hparams.optimizer_config)

        # Predictor parameters have a 10x learning rate.
        backbone_params, predictor_params = [], []
        for name, param in self.named_parameters():
            if "predictor" in name:
                predictor_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.SGD(
            [
                {"params": backbone_params, "lr": config["lr"]},
                {"params": predictor_params, "lr": config["lr"] * 10},
            ],
            momentum=config["momentum"],
            weight_decay=config["wd"],
            nesterov=config["nesterov_use"],
        )
        # if config["lars_use"]:
        #     optimizer = pl_bolts.optimizers.lars_scheduling.LARSWrapper(
        #         optimizer, eta=config["trust_coefficient"]
        #     )

        if config["scheduler_name"] == "cosine_decay":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config["scheduler_warmup_steps"],
                last_epoch=config["scheduler_max_epochs"],
            )
            return optimizer, lr_scheduler

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--arch", default="r2plus1d_18", choices=["mc3_18", "r2plus1d_18", "r3d_18"]
        )
        parser.add_argument("--num_features", default=2048, type=int)
        parser.add_argument("--output_dim", default=128, type=int)
        parser.add_argument("--optimizer_lr", default=4.8, type=float)
        parser.add_argument("--optimizer_momentum", default=0.9, type=float)
        parser.add_argument("--optimizer_wd", default=0.0000001, type=float)
        parser.add_argument("--optimizer_nesterov_use", default=True, type=bool)
        parser.add_argument("--optimizer_lars_use", default=True, type=bool)
        parser.add_argument(
            "--optimizer_lars_trust_coefficient", default=0.001, type=float
        )
        parser.add_argument(
            "--optimizer_scheduler_name", default="cosine-decay", type=str
        )
        parser.add_argument(
            "--optimizer_scheduler_warmup_steps", default=5000, type=int
        )
        parser.add_argument(
            "--optimizer_scheduler_max_epochs", default=300000, type=int
        )
        return parser
