import os
from typing import Optional, Union

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]


class PlanetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_folder: str,
        transforms: Optional[TRANSFORM_TYPE] = None,
    ):
        self.df = df
        self.image_folder = image_folder
        self.transforms = transforms

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_folder, f'{row.Id}.jpg')
        labels = np.array(row.drop(['Id']), dtype='float32')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = {'image': image, 'labels': labels}

        if self.transforms:
            data = self.transforms(**data)

        return data['image'], data['labels']

    def __len__(self) -> int:
        return len(self.df)
