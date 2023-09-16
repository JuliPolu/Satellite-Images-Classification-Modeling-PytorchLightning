import typing as tp
import os
import cv2
import numpy as np
import pandas as pd
import torch
from src.lightning_module import PlanetModule
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def preprocess_image(image: np.ndarray, target_image_size: tp.Tuple[int, int]) -> torch.Tensor:
    """Препроцессинг имаджнетом.

    :param image: RGB-изображение;
    :param target_image_size: целевой размер изображения;
    :return: обработанный тензор.
    """
    image = image.astype(np.float32)

    preprocess = albu.Compose(
            [
                albu.Resize(height=target_image_size, width=target_image_size),
                albu.Normalize(),
                ToTensorV2(),
            ]
    )

    image = preprocess(image=image)['image']

    return image

    # return torch.from_numpy(image)


DATA_FOLDER = './data/'

if __name__ == '__main__':
    checkpoint_name = './models/eff-b0_base_aug/epoch_epoch=11-val_f1=0.668.ckpt'
    model = PlanetModule.load_from_checkpoint(checkpoint_name, map_location=torch.device('cpu'))
    img = cv2.imread('data/test-jpg/test_0.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img, 224)

    df = pd.read_csv(os.path.join(DATA_FOLDER, 'df_test.csv'))
    names = list(df.columns[1:])

    with torch.no_grad():
        # probs = model(img[None]).detach().cpu().numpy()[0]
        probs = torch.sigmoid(model(img[None]))[0].detach().cpu().numpy()
    class_name2prob = {cls: prob for cls, prob in zip(names, probs)}
    print(class_name2prob)