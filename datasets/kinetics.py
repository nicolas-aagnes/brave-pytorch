from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset
import os
import torch

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from typing import Optional
from dataclasses import dataclass
import torchvision.transforms._transforms_video as transforms_video

import dataset
import numpy as np


class Kinetics(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.
    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.
    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.
    Internally, it uses a VideoClips object to handle clip creation.
    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

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

        self.clips_start_index = 0
        self.clips_end_index = self.video_clips.num_clips()  # Non-inclusive index

    def set_clips_start_and_end_indices(self, start_index, end_index):
        self.clips_start_index = start_index
        self.clips_end_index = end_index

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.clips_end_index - self.clips_start_index

    def __getitem__(self, idx):
        index = idx + self.clips_start_index
        # video_q, audio_q, info_q, video_idx_q = self.video_clips.get_clip(idx[0])
        # video_k, audio_k, info_k, video_idx_k = self.video_clips.get_clip(idx[1])
        video, audio, info, video_idx = self.video_clips.get_clip(index)

        video = tuple(transform["video"](video) for transform in self.transforms)
        audio = tuple(transform["audio"](audio) for transform in self.transforms)

        return video, audio


class KineticsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, frames_per_clip: int, frame_rate: int = 25):
        super().__init__()

        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate

        def get_transform(crop_size, num_frames, step):
            video_transformation = transforms.Compose(
                [
                    transforms_video.ToTensorVideo(),
                    transforms_video.RandomResizedCropVideo(crop_size, (0.2, 1)),
                ]
            )
            audio_transformation = dataset.DummyAudioTransform()
            transformation = {
                "video": video_transformation,
                "audio": audio_transformation,
            }
            self.transforms.append(transformation)

        self.transforms = [get_transform(112, 16, 2), get_transform(224, 64, 4)]

    def prepare_data(self):
        # Create video clips.
        self.kinetics = Kinetics(
            root=self.data_dir,
            view_configs=self.view_configs,
            frames_per_clip=self.frames_per_clip,
            step_between_clips=1,
            frame_rate=self.frame_rate,
        )

        # Split data into train, val and test.
        num_clips = len(self.kinetics)
        self.indices = np.random.permutation(num_clips)
        self.indices_train = self.indices[: int(num_clips * 0.6)]
        self.indices_val = self.indices[int(num_clips * 0.6) : int(num_clips * 0.8)]
        self.indices_test = self.indices[int(num_clips * 0.8) :]

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.dataset_train = self.kinetics.subset(self.indices_train)
            self.dataset_val = self.kinetics.subset(self.indices_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.dataset_test = self.kinetics.subset(self.indicest_test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
