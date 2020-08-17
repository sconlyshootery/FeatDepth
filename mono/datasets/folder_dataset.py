from __future__ import absolute_import, division, print_function
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import os

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(filename):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(filename, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class FolderDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 is_train=False,
                 img_ext='.jpg',
                 gt_depth_path = None):
        super(FolderDataset, self).__init__()

        self.data_path = data_path
        self.filenames = sorted(os.listdir(data_path))
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS
        self.is_train = is_train
        self.frame_idxs = frame_idxs
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.K = np.array([[0.9765, 0, 0.5, 0],
                           [0, 1.736, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # Need to specify augmentations differently in pytorch 1.0 compared with 0.4
        if int(torch.__version__.split('.')[0]) > 0:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        else:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)

        self.flag = np.zeros(self.__len__(), dtype=np.int64)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                inputs[(n, im, 0)] = self.resize(inputs[(n, im, - 1)])

        for k in list(inputs):
            if "color" in k:
                f = inputs[k]
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if i == 0:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        for i in self.frame_idxs:
            try:
                filename = self.filenames[index+i]
            except:
                filename = self.filenames[index]

            inputs[("color", i, -1)] = self.get_color(filename, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height
        inv_K = np.linalg.pinv(K)

        inputs[("K", 0)] = torch.from_numpy(K)
        inputs[("inv_K", 0)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]

        return inputs

    def get_color(self, filename, do_flip):
        color = self.loader(os.path.join(self.data_path, filename))

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color