import glob
import os
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset

from datasets import video_transforms


class HMDB51(VisionDataset):
    """
    `HMDB51 <http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_
    dataset.

    HMDB51 is an action recognition video dataset.
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
        root (string): Root directory of the HMDB51 Dataset.
        annotation_path (str): Path to the folder containing the split files.
        frames_per_clip (int): Number of frames in a clip.
        step_between_clips (int): Number of frames between each clip.
        fold (int, optional): Which fold to use. Should be between 1 and 3.
        train (bool, optional): If ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C]): The `T` video frames
            - audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
              and `L` is the number of points
            - label (int): class of the video clip
    """

    data_url = (
        "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    )
    splits = {
        "url": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
        "md5": "15e67781e70dcfbdce2d7dbb9b3344b5",
    }
    TRAIN_TAG = 1
    TEST_TAG = 2

    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        fold: int = 1,
        train: bool = True,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
    ) -> None:
        super().__init__(root)
        if fold not in (1, 2, 3):
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        extensions = ("avi",)
        self.classes, class_to_idx = find_classes(self.root)
        print(len(self.classes))
        self.samples = make_dataset(
            self.root,
            class_to_idx,
            extensions,
        )
        # print(self.samples)

        video_paths = [path for (path, _) in self.samples]
        video_clips = VideoClips(
            video_paths,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        # we bookkeep the full version of video clips because we want to be able
        # to return the meta data of full version rather than the subset version of
        # video clips
        self.full_video_clips = video_clips
        self.fold = fold
        self.train = train
        self.indices = self._select_fold(video_paths, annotation_path, fold, train)
        print(len(self.indices))
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.full_video_clips.metadata

    def _select_fold(
        self, video_list: List[str], annotations_dir: str, fold: int, train: bool
    ) -> List[int]:
        target_tag = self.TRAIN_TAG if train else self.TEST_TAG
        split_pattern_name = "*test_split{}.txt".format(fold)
        split_pattern_path = os.path.join(annotations_dir, split_pattern_name)
        annotation_paths = glob.glob(split_pattern_path)
        selected_files = set()
        for filepath in annotation_paths:
            with open(filepath) as fid:
                lines = fid.readlines()
            for line in lines:
                video_filename, tag_string = line.split()
                tag = int(tag_string)
                if tag == target_tag:
                    selected_files.add(video_filename)

        indices = []
        for video_index, video_path in enumerate(video_list):
            if os.path.basename(video_path) in selected_files:
                indices.append(video_index)

        return indices

    def __len__(self) -> int:
        return 1000
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        idx = int(np.random.choice(len(self), 1)[0])
        while True:
            try:
                video, audio, _, video_idx = self.video_clips.get_clip(idx)
                break
            except AssertionError as e:
                pass

            idx = int(np.random.choice(len(self), 1)[0])

        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        return video, class_index


class HMDB51DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        annotation_path,
        data_fold,
        num_workers,
        batch_size,
        image_size,
        num_frames,
        frame_step,
    ):
        super().__init__()
        self.save_hyperparameters()

        transform = video_transforms.VideoTransformTrain(
            image_size, num_frames, frame_step
        )

        metadata_path_train = os.path.join(
            data_dir, f"kinetics_metadata_train_fold{data_fold}.pt"
        )
        metadata_path_test = os.path.join(
            data_dir, f"kinetics_metadata_test_fold{data_fold}.pt"
        )

        metadata_train = (
            torch.load(metadata_path_train)
            if os.path.exists(metadata_path_train)
            else None
        )
        metadata_test = (
            torch.load(metadata_path_test)
            if os.path.exists(metadata_path_test)
            else None
        )

        self.hmdb_train = HMDB51(
            root=data_dir,
            annotation_path=annotation_path,
            frames_per_clip=num_frames * frame_step,
            fold=data_fold,
            train=True,
            transform=transform,
            _precomputed_metadata=metadata_train,
        )
        self.hmdb_test = HMDB51(
            root=data_dir,
            annotation_path=annotation_path,
            frames_per_clip=num_frames * frame_step,
            fold=data_fold,
            train=False,
            transform=transform,
            _precomputed_metadata=metadata_test,
        )

        if not os.path.exists(metadata_path_train):
            torch.save(self.hmdb_train.metadata, metadata_path_train)
        if not os.path.exists(metadata_path_test):
            torch.save(self.hmdb_test.metadata, metadata_path_test)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset = self.hmdb_train

        if stage == "test" or stage is None:
            self.dataset = self.hmdb_test

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", required=True, type=str)
        parser.add_argument("--annotation_path", required=True, type=str)
        parser.add_argument("--data_fold", default=1, choices=[1, 2, 3], type=int)
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--image_size", default=224, type=int)
        parser.add_argument("--num_frames", default=32, type=int)
        parser.add_argument("--frame_step", default=2, type=int)
        return parser
