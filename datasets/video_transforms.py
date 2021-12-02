import random

import kornia
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips


class PermuteVideoChannels(object):
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, video_clip):
        return torch.permute(video_clip, self.permutation)


class SelectFrames(object):
    """Select frames at set step."""

    def __init__(self, num_frames, step, random_start):
        self.num_frames = num_frames
        self.step = step
        self.random_start = random_start

    def __call__(self, video):
        # Video should be of shape (T x H x W x C)
        assert len(video.shape) == 4 and video.shape[-1] == 3
        if self.random_start:
            start_index = random.randint(
                0, video.shape[0] - (self.num_frames * self.step) - 1
            )
            return video[
                range(start_index, start_index + self.num_frames * self.step, self.step)
            ]
        else:
            return video[range(0, self.num_frames * self.step, self.step)]


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0], img_size=112):
        self.sigma = sigma
        self.radius = int(0.1 * img_size) // 2 * 2 + 1

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        gauss = kornia.filters.GaussianBlur2d(
            (self.radius, self.radius), (sigma, sigma)
        )
        return gauss(x)


class Scale(object):
    def __call__(self, x):
        # assert x.max() > 1.0
        assert -0.1 <= x.min() <= x.max() <= 255.1, (
            x.shape,
            x.min(),
            x.max(),
            x.mean(),
        )
        return x / 255.0


class VideoTransformTrain(object):
    def __init__(self, crop_size, num_frames, step, random_start):

        self.transform = transforms.Compose(
            [
                SelectFrames(num_frames, step, random_start),
                PermuteVideoChannels((0, 3, 1, 2)),  # T x H x W x C ---> T x C x H x W
                transforms.RandomResizedCrop(
                    crop_size, scale=(0.3, 1), ratio=(0.5, 2.0)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],  # not strengthened
                #     p=0.8,
                # ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0], crop_size)], p=0.5),
                Scale(),
                transforms.Normalize(
                    mean=torch.tensor([0.45, 0.45, 0.45]),
                    std=torch.tensor([0.225, 0.225, 0.225]),
                ),
                PermuteVideoChannels((1, 0, 2, 3)),  # T x C x H x W ---> C x T x H x W
            ]
        )

    def __call__(self, x):
        # Input tensor should be of shape T x H x W x C.
        x = self.transform(x.float())
        return x


class DummyAudioTransform(object):
    """This is a dummy audio transform.
    It ignores actual audio data, and returns an empty tensor. It is useful when
    actual audio data is raw waveform and has a varying number of waveform samples
    which makes minibatch assembling impossible
    """

    def __init__(self):
        pass

    def __call__(self, _audio):
        return torch.zeros(0, 1, dtype=torch.float)
