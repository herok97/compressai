# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        # print(self.samples)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        try:
            img = Image.open(self.samples[index]).convert("RGB")
            if self.transform:
                # print(type(self.transform(img)))
                return self.transform(img)
            return img
        except:
            pass

    def __len__(self):
        return len(self.samples)


class VideoFolder(Dataset):

    def __init__(self, root, mode='train', transform=None):
        from tqdm import tqdm
        self.mode = mode
        root_dir = Path(root)
        self.transform = transform

        if self.mode == 'train':
            from random import sample
            self.samples = []
            for sub_f in tqdm(root_dir.iterdir()):
                if sub_f.is_dir():
                    for sub_sub_f in Path(sub_f).iterdir():
                        for i in range(5):
                            # print(sample(list(sub_sub_f.iterdir()), k=2))
                            self.samples.append(sample(list(sub_sub_f.iterdir()), k=2))

            if not root_dir.is_dir():
                raise RuntimeError(f'Invalid directory "{root}"')

            # self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        else:
            self.samples = {}
            self.video_names = []
            for sub_f in tqdm(root_dir.iterdir()):
                if sub_f.is_dir():
                    self.video_names.append(Path(sub_f).name)
                    self.samples[Path(sub_f).name] = [img for img in Path(sub_f).iterdir() if img.is_file()]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        if self.mode == 'train':
            try:
                imgs = self.samples[index]
                img1 = Image.open(imgs[0]).convert("RGB")
                img2 = Image.open(imgs[1]).convert("RGB")
                if self.transform:
                    return [self.transform(img1), self.transform(img2)]
                else:
                    return [img1, img2]
            except:
                pass
        # test
        else:
            video_name = self.video_names[index]
            imgs = self.samples[video_name]
            if self.transform:
                return (video_name, [self.transform(Image.open(img).convert("RGB")) for img in imgs])
            else:
                return imgs

    def __len__(self):
        return len(self.samples)
