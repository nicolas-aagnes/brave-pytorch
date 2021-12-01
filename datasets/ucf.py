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


class UCF101(VisionDataset):
    """
    `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    UCF101 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``. The dataset itself can be downloaded from the dataset website;
    annotations that ``annotation_path`` should be pointing to can be downloaded from `here
    <https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip>`.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the UCF101 Dataset.
        annotation_path (str): path to the folder containing the split files;
            see docstring above for download instructions of these files
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        fold (int, optional): which fold to use. Should be between 1 and 3.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        tuple: A 3-tuple with the following entries:

            - video (Tensor[T, H, W, C]): the `T` video frames
            -  audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
               and `L` is the number of points
            - label (int): class of the video clip
    """

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
        if not 1 <= fold <= 3:
            raise ValueError(f"fold should be between 1 and 3, got {fold}")

        extensions = ("avi",)
        self.fold = fold
        self.train = train

        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(
            self.root, class_to_idx, extensions, is_valid_file=None
        )
        video_list = [x[0] for x in self.samples]
        video_clips = VideoClips(
            video_list,
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
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.full_video_clips.metadata

    def _select_fold(
        self, video_list: List[str], annotation_path: str, fold: int, train: bool
    ) -> List[int]:
        name = "train" if train else "test"
        name = f"{name}list{fold:02d}.txt"
        f = os.path.join(annotation_path, name)
        selected_files = set()
        with open(f) as fid:
            data = fid.readlines()
            data = [x.strip().split(" ")[0] for x in data]
            data = [os.path.join(self.root, x) for x in data]
            selected_files.update(data)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self) -> int:
        return 10000
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        idx = int(np.random.choice(self.video_clips.num_clips(), 1)[0])
        while True:
            try:
                video, audio, _, video_idx = self.video_clips.get_clip(idx)
                break
            except AssertionError as e:
                pass

            idx = int(np.random.choice(self.video_clips.num_clips(), 1)[0])

        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label


class UCF101DataModule(pl.LightningDataModule):
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
            image_size, num_frames, frame_step, random_start=False
        )

        metadata_path_train = os.path.join(
            "./data/ucf101", f"ucf_metadata_train_fold{data_fold}.pt"
        )
        metadata_path_test = os.path.join(
            "./data/ucf101", f"ucf_metadata_test_fold{data_fold}.pt"
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

        self.ucf_train = UCF101(
            root=data_dir,
            annotation_path=annotation_path,
            frames_per_clip=num_frames * frame_step,
            fold=data_fold,
            train=True,
            transform=transform,
            _precomputed_metadata=metadata_train,
            num_workers=32,
        )
        self.ucf_test = UCF101(
            root=data_dir,
            annotation_path=annotation_path,
            frames_per_clip=num_frames * frame_step,
            fold=data_fold,
            train=False,
            transform=transform,
            _precomputed_metadata=metadata_test,
            num_workers=32,
        )

        print("train len", len(self.ucf_train))
        print("len test", len(self.ucf_test))

        if not os.path.exists(metadata_path_train):
            torch.save(self.ucf_train.metadata, metadata_path_train)
        if not os.path.exists(metadata_path_test):
            torch.save(self.ucf_test.metadata, metadata_path_test)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset = self.ucf_train

        if stage == "test" or stage is None:
            self.dataset = self.ucf_test

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", required=True, type=str)
        parser.add_argument("--annotation_path", required=True, type=str)
        parser.add_argument("--data_fold", default=1, choices=[1, 2, 3], type=int)
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--image_size", default=224, type=int)
        parser.add_argument("--num_frames", default=32, type=int)
        parser.add_argument("--frame_step", default=2, type=int)
        return parser
