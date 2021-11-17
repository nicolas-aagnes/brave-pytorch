import random

import cv2
import kornia
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips


class VideoClipToTensor(object):
    """
    change input channel
    D x H x W x C ---> C x D x H x w
    """

    def __call__(self, video_clip):
        return np.transpose(video_clip, (3, 0, 1, 2))


class SelectFrames(object):
    """Select frames at set step."""

    def __init__(self, num_frames, step):
        self.num_frames = num_frames
        self.step = step

    def __call__(self, video):
        # Video should be of shape (T, C, H, W)
        print("got video shape", video.shape)
        assert len(video.shape) == 4 and video.shape[1] == 3
        assert self.num_frames * self.step <= video.shape[0]
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


class MoCoAugmentV2(object):
    def __init__(self, crop_size):

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        self.moco_augment_v2 = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        kornia.augmentation.ColorJitter(
                            0.4, 0.4, 0.4, 0.1
                        )  # not strengthened
                    ],
                    p=0.8,
                ),
                kornia.augmentation.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0], crop_size)], p=0.5),
                kornia.augmentation.RandomHorizontalFlip(),
                normalize_video,
            ]
        )

    def __call__(self, clips):
        # from (B, C, T, H, W) to (B, T, C, H, W)
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()
        clips_batch = clips.view(-1, clips.shape[2], clips.shape[3], clips.shape[4])
        aug_clips = self.moco_augment_v2(clips_batch)
        aug_clips = aug_clips.view(clips.shape)
        # from (B, T, C, H, W) to (B, C, T, H, W)
        aug_clips = aug_clips.permute(0, 2, 1, 3, 4).contiguous()
        return aug_clips


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
