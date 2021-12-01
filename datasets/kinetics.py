import os
import warnings
from argparse import ArgumentParser
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.samplers.clip_sampler import (DistributedSampler,
                                                        RandomClipSampler)
from torchvision.datasets.utils import list_dir
from torchvision.datasets.video_utils import VideoClips

from datasets import video_transforms

warnings.filterwarnings("ignore")


class KineticsDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root,
        frames_per_clip,
        step_between_clips=1,
        frame_rate=None,
        extensions=("avi",),
        transforms=None,
        num_workers=1,
        _video_width=0,
        _video_height=0,
        _video_min_dimension=0,
        _audio_samples=0,
    ):
        super().__init__(root)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(
            self.root, class_to_idx, extensions, is_valid_file=None
        )
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_list = video_list
        split = root.split("/")[-1].strip("/")
        metadata_filepath = os.path.join(root, "kinetics_metadata_{}.pt".format(split))

        if os.path.exists(metadata_filepath):
            metadata = torch.load(metadata_filepath)
        else:
            metadata = None

        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        self.transforms = transforms

        if not os.path.exists(metadata_filepath):
            torch.save(self.video_clips.metadata, metadata_filepath)

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, index):
        while True:
            try:
                video, _, info, video_idx = self.video_clips.get_clip(index)
                break
            except AssertionError as e:
                pass

            index = int(np.random.choice(len(self), 1)[0])

        videos = tuple(
            transform["video"](video.clone()) for transform in self.transforms
        )

        return videos


class KineticsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        num_workers,
        batch_size,
        frame_rate,
        image_size_student,
        num_frames_student,
        step_student,
        image_size_teacher,
        num_frames_teacher,
        step_teacher,
    ):
        super().__init__()
        self.save_hyperparameters()

        def get_transform(crop_size, num_frames, step, random_start):
            return {
                "video": video_transforms.VideoTransformTrain(
                    crop_size, num_frames, step, random_start
                ),
                "audio": video_transforms.DummyAudioTransform(),
            }

        self.transforms = [
            get_transform(image_size_student, num_frames_student, step_student, True),
            get_transform(image_size_teacher, num_frames_teacher, step_teacher, False),
        ]

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.dataset_train = KineticsDataset(
                root=self.hparams.data_dir,
                frames_per_clip=max(
                    self.hparams.num_frames_student * self.hparams.step_student,
                    self.hparams.num_frames_teacher * self.hparams.step_teacher,
                ),
                step_between_clips=1,
                frame_rate=self.hparams.frame_rate,
                transforms=self.transforms,
                extensions="mp4",
                num_workers=4,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            raise NotImplementedError

    def train_dataloader(self):
        train_sampler = RandomClipSampler(self.dataset_train.video_clips, 1)
        # train_sampler = DistributedSampler(train_sampler)
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=True,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            multiprocessing_context="fork",
            pin_memory=True,
            persistent_workers=True,
        )

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", required=True, type=str)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--frame_rate", default=30, type=int)
        parser.add_argument("--image_size_student", default=224, type=int)
        parser.add_argument("--num_frames_student", default=16, type=int)
        parser.add_argument("--step_student", default=2, type=int)
        parser.add_argument("--image_size_teacher", default=112, type=int)
        parser.add_argument("--num_frames_teacher", default=64, type=int)
        parser.add_argument("--step_teacher", default=4, type=int)
        return parser
