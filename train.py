"""Example launcher for a hyperparameter search on SLURM.

This example shows how to use gpus on SLURM with PyTorch.
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster

from datasets.kinetics import KineticsDataModule
from datasets.kinetics_dummy import DummyKineticsDataModule
from models.brave import Brave


def train(args, cluster):
    print(args)

    model = Brave(
        args.arch,
        args.num_features,
        args.hidden_dim,
        args.output_dim,
        args.learning_rate,
    )

    datamodule = (
        KineticsDataModule.from_argparse_args(args)
        if not args.dummy_data
        else DummyKineticsDataModule.from_argparse_args(args)
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="loss", filename="model-{epoch:02d}", save_top_k=5
    )
    # early_stop_callback = EarlyStopping(
    #     monitor="loss", min_delta=0.00, patience=6, verbose=True, mode="min"
    # )

    path = Path(args.test_tube_slurm_cmd_path)
    save_folder, file_name = path.parent.parent, path.name
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger(
            save_dir=save_folder / "logs" / file_name.replace(".sh", ""),
            default_hp_metric=False,
        ),
        replace_sampler_ddp=False,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy="random_search")
    parser.add_argument("--test_tube_exp_name", default="brave")
    parser.add_argument("--log_path", default="./runs_clean")
    parser.add_argument("--dummy_data", action="store_true")
    parser = KineticsDataModule.add_model_specific_args(parser)
    parser = Brave.add_model_specific_args(parser, hyperopt=True)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=args,
        log_path=args.log_path,
        python_cmd="python",
    )

    # Add commands to the non-SLURM portion.
    cluster.add_command("cd /vision/u/naagnes/github/brave-pytorch")
    cluster.add_command("source .svl/bin/activate")

    # SLURM commands.
    cluster.add_slurm_cmd(cmd="partition", value="macondo", comment="")
    cluster.add_slurm_cmd(cmd="qos", value="normal", comment="")
    cluster.add_slurm_cmd(cmd="time", value="48:00:00", comment="")
    cluster.add_slurm_cmd(cmd="ntasks-per-node", value=1, comment="")
    cluster.add_slurm_cmd(cmd="cpus-per-task", value=32, comment="")
    cluster.add_slurm_cmd(cmd="mem", value="120G", comment="")

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 4
    cluster.per_experiment_nb_nodes = 1
    cluster.gpu_type = "titanrtx"

    # Each hyperparameter combination will use 8 gpus.
    cluster.optimize_parallel_cluster_gpu(
        # Function to execute:
        train,
        # Number of hyperparameter combinations to search:
        nb_trials=2,
        # This is what will display in the slurm queue:
        job_name="brave",
    )
