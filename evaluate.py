"""Example launcher for a hyperparameter search on SLURM.

This example shows how to use gpus on SLURM with PyTorch.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
from sklearn import svm
from test_tube import HyperOptArgumentParser, SlurmCluster

from datasets.ucf import UCF101DataModule
from models.brave import Brave


@dataclass
class ClassificationResults:
    top_one_accuracy: float
    top_five_accuracy: float


class EvaluationResults(NamedTuple):
    test: ClassificationResults
    train: ClassificationResults


def _compute_accuracy_metrics(labels, predictions) -> ClassificationResults:
    """Compute accuracy metrics."""
    labels = labels[..., None]

    sorted_predictions = np.argsort(predictions, axis=1)
    assert len(labels.shape) == len(sorted_predictions.shape) == 2

    top1_predictions = sorted_predictions[:, -1:]
    top5_predictions = sorted_predictions[:, -5:]

    return ClassificationResults(
        top_one_accuracy=np.mean(top1_predictions == labels),
        top_five_accuracy=np.mean(np.max(top5_predictions == labels, 1)),
    )


def evaluate(args, cluster):
    print(args)

    model = Brave.load_from_checkpoint(args.checkpoint_path)
    datamodule = UCF101DataModule.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args)

    print("Computing training embeddings.")
    datamodule.setup("fit")
    out = trainer.predict(model, datamodule=datamodule)
    train_embeddings = torch.cat(list(x for x, y in out)).detach().cpu().numpy()
    train_labels = torch.cat(list(y for x, y in out)).detach().cpu().numpy()
    print(np.unique(train_labels))

    print("Computing test embeddings.")
    datamodule.setup("test")
    out = trainer.predict(model, datamodule=datamodule)
    test_embeddings = torch.cat(list(x for x, y in out)).detach().cpu().numpy()
    test_labels = torch.cat(list(y for x, y in out)).detach().cpu().numpy()

    print("Train and test shapes:")
    print(
        train_embeddings.shape,
        train_labels.shape,
        test_embeddings.shape,
        test_labels.shape,
    )

    print("Fitting scaler.")
    start = time.time()
    scaler = sklearn.preprocessing.StandardScaler().fit(train_embeddings)
    print("Fitting scaler took", time.time() - start)

    print("Rescaling features.")
    start = time.time()
    train_embeddings = scaler.transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)
    print(
        "Rescaling features took",
        time.time() - start,
        "with shape",
        train_embeddings.shape,
        test_embeddings.shape,
    )

    svm_regularization = 0.0001
    print("Fitting an SVM with regularization", svm_regularization)
    start = time.time()
    classifier = svm.LinearSVC(C=svm_regularization)
    classifier.fit(train_embeddings, train_labels)
    print("SVM fitting took", time.time() - start)

    print("Computing predictions.")
    start = time.time()
    train_predictions = classifier.decision_function(train_embeddings)
    test_predictions = classifier.decision_function(test_embeddings)
    print(train_predictions.shape, test_predictions.shape)
    print("Computing predictions took", time.time() - start)

    results = EvaluationResults(
        test=_compute_accuracy_metrics(test_labels, test_predictions),
        train=_compute_accuracy_metrics(train_labels, train_predictions),
    )
    print(results)


if __name__ == "__main__":
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy="grid_search")
    parser.add_argument("--test_tube_exp_name", default="ucf101")
    parser.add_argument("--log_path", default="./evaluation")
    parser.opt_list(
        "--checkpoint_path",
        options=[
            "/vision/u/naagnes/github/brave-pytorch/runs_clean/brave/logs/trial_5_2021-11-30__19-57-44_slurm_cmd/default/version_0/checkpoints/model-epoch=40.ckpt",
            "/vision/u/naagnes/github/brave-pytorch/runs_clean/brave/logs/trial_7_2021-11-30__19-57-44_slurm_cmd/default/version_0/checkpoints/model-epoch=46.ckpt",
        ],
        type=str,
        tunable=True,
    )
    parser = UCF101DataModule.add_model_specific_args(parser)
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
    cluster.add_slurm_cmd(cmd="partition", value="svl", comment="")
    cluster.add_slurm_cmd(cmd="qos", value="normal", comment="")
    cluster.add_slurm_cmd(cmd="time", value="24:00:00", comment="")
    cluster.add_slurm_cmd(cmd="ntasks-per-node", value=1, comment="")
    cluster.add_slurm_cmd(cmd="cpus-per-task", value=32, comment="")
    cluster.add_slurm_cmd(cmd="mem", value="30G", comment="")

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.gpu_type = "titanrtx"

    cluster.optimize_parallel_cluster_gpu(evaluate, nb_trials=2, job_name="ucf101")
