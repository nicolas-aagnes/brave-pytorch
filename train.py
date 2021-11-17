from argparse import ArgumentParser
from typing import Any, Dict

import pytorch_lightning as pl

from datasets.kinetics import KineticsDataModule
from datasets.kinetics_dummy import DummyKineticsDataModule
from models.brave import Brave


def main():
    pl.seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument("--dummy_data", action="store_true")
    parser = KineticsDataModule.add_model_specific_args(parser)
    parser = Brave.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    optimizer_config = {
        k.replace("optimizer_", ""): v
        for k, v in dict(vars(args)).items()
        if "optimizer" in k
    }
    model = Brave(args.arch, args.num_features, args.output_dim, optimizer_config)

    datamodule = (
        KineticsDataModule.from_argparse_args(args)
        if not args.dummy_data
        else DummyKineticsDataModule.from_argparse_args(args)
    )

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = pl.loggers.TensorBoardLogger(
        save_dir="runs", name="brave", default_hp_metric=False
    )
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
