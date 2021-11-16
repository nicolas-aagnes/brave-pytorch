from argparse import ArgumentParser
from typing import Any, ForwardRef

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.nn.functional as F

from pl_bolts.callbacks import (
    LatentDimInterpolator,
    TensorboardGenerativeModelImageSampler,
)
from pl_bolts.models.gans.dcgan.components import DCGANDiscriminator, DCGANGenerator
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import LSUN, MNIST
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


video_model_names = sorted(
    name
    for name in torchvision.models.video.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision.models.video.__dict__[name])
)


class Projector(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_features: int = 4096
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, x):
        return self.projector(x)


class Predictor(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, hidden_features: int = 4096
    ):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.predictor(x)


class Brave(pl.LightningModule):
    def __init__(self, num_features: int = 2048):
        super().__init__(self)

        # This saves all of the arguments to __init__ in an hparams dictionary.
        self.save_hyperparameters(ignore=["student", "teacher"])
        # self.hparams.update(student.hparams)
        # self.hparams.update(teacher.hparams)

        self.student = video_model_names["r3d_18"](num_classes=num_features)
        self.teacher = video_model_names["r3d_18"](num_classes=num_features)
        self.projector = Projector(num_features, num_features)
        self.predictor = Predictor(num_features, num_features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, batch):
        # Get student and teacher batch.
        x_s, x_t = batch

        # Compute student and teacher embeddings.
        f_s = self.student(x_s)
        f_t = self.teacher(x_t)

        # Compute projected embeddings.
        z_s = self.projector(f_s)
        z_t = self.projectot(f_t)

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

    def training_step(self, batch, batch_idx):
        return self(batch)

    def validation_step(self, batch, batch_idx):
        return self(batch)


class DummyData(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return torch.zeros(16, 10, 3, 224, 224), torch.zeros(16, 10, 3, 224, 224)

    def __len__(self):
        return 1


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = DummyData()
    train_loader = DataLoader(dataset, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    student = torchvision.models.efficientnet_b0(num_classes=args.num_features)
    teacher = torchvision.models.efficientnet_b0(num_classes=args.num_features)
    model = Brave(student, teacher, args.num_features)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader)

    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)


if __name__ == "__main__":
    cli_main()
