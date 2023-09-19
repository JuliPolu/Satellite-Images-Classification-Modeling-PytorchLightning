import logging
import os
from typing import Optional, Tuple

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms
from src.config import DataConfig
from src.dataset import PlanetDataset
from src.dataset_splitter import stratify_shuffle_split_subsets


class PlanetDM(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self._config = config
        self._train_transforms = get_transforms(width=config.width, height=config.height)
        self._valid_transforms = get_transforms(width=config.width, height=config.height, augmentations=False)
        self._image_folder = os.path.join(config.data_path, 'Images')

        self.label_names = []
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self):
        self.label_names = split_and_save_datasets(self._config.data_path, self._config.train_size)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            df_train = read_df(self._config.data_path, 'train')
            df_valid = read_df(self._config.data_path, 'valid')
            self.train_dataset = PlanetDataset(
                df_train,
                image_folder=self._image_folder,
                transforms=self._train_transforms,
            )
            self.valid_dataset = PlanetDataset(
                df_valid,
                image_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

        elif stage == 'test':
            df_test = read_df(self._config.data_path, 'test')
            self.test_dataset = PlanetDataset(
                df_test,
                image_folder=self._image_folder,
                transforms=self._valid_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )


def split_and_save_datasets(data_path: str, train_fraction: float = 0.8):
    df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    logging.info(f'Original dataset: {len(df)}')
    label_names = list(df.columns[2:])
    df = df.drop_duplicates()
    df = df.drop(['tags_list'], axis=1)
    logging.info(f'Final dataset: {len(df)}')

    train_df, valid_df, test_df = stratify_shuffle_split_subsets(df, train_fraction=train_fraction)
    logging.info(f'Train dataset: {len(train_df)}')
    logging.info(f'Valid dataset: {len(valid_df)}')
    logging.info(f'Test dataset: {len(test_df)}')

    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)
    logging.info('Datasets successfully saved!')
    return label_names


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
