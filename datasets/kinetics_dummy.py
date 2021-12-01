from argparse import ArgumentParser
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torchvision

from datasets import video_transforms


class DummyKineticsDataset(torchvision.datasets.VisionDataset):
    def __init__(self, num_clips: int, params: Dict[str, int]):
        self.num_clips = num_clips
        self.params = params

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        student_view = torch.zeros(
            3,
            self.params["num_frames_student"],
            self.params["image_size_student"],
            self.params["image_size_student"],
        )
        teacher_view = torch.zeros(
            3,
            self.params["num_frames_teacher"],
            self.params["image_size_teacher"],
            self.params["image_size_teacher"],
        )
        return student_view, teacher_view


class DummyKineticsDataModule(pl.LightningDataModule):
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

        def get_transform(crop_size, num_frames, step):
            video_transformation = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.RandomResizedCrop(crop_size, (0.2, 1)),
                ]
            )
            audio_transformation = video_transforms.DummyAudioTransform()
            transformation = {
                "video": video_transformation,
                "audio": audio_transformation,
            }
            return transformation

        self.transforms = [
            get_transform(image_size_student, num_frames_student, step_student),
            get_transform(image_size_teacher, num_frames_teacher, step_teacher),
        ]

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.dataset_train = DummyKineticsDataset(20000, dict(self.hparams))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            raise NotImplementedError

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", default="", type=str)
        parser.add_argument("--num_workers", default=0, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--frame_rate", default=25, type=int)
        parser.add_argument("--image_size_student", default=224, type=int)
        parser.add_argument("--num_frames_student", default=16, type=int)
        parser.add_argument("--step_student", default=2, type=int)
        parser.add_argument("--image_size_teacher", default=112, type=int)
        parser.add_argument("--num_frames_teacher", default=64, type=int)
        parser.add_argument("--step_teacher", default=4, type=int)
        return parser
