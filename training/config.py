"""Configuration for the Brave experiment."""

import glob
import ml_collections

from brave.models.brave import config as brave_config


def get_config() -> ml_collections.ConfigDict:
    """Get the experiment config."""

    config = ml_collections.ConfigDict()

    config.checkpoint_dir = "/tmp/jaxline/brave"
    config.train_checkpoint_all_hosts = False
    config.training_steps = 300_000
    config.log_tensors_interval = 60
    config.save_checkpoint_interval = 600
    config.eval_specific_checkpoint_dir = ""
    config.best_model_eval_metric = "multiple_of_saving_period"

    config.experiment_kwargs.experiment_name = "brave"
    config.experiment_kwargs.config = brave_config.get_experiment_config()
    config.eval_modes = config.experiment_kwargs.config.eval_modes

    # Fill in this to set the training shards for training.
    config.experiment_kwargs.config.model.dataset_shards = glob.glob(
        "<path/to/train/shards/*.tfrecord>"
    )

    config.lock()
    return config
