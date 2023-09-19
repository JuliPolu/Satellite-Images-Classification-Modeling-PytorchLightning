import typing as tp
import os
import cv2
import numpy as np
import pandas as pd
import torch
from src.lightning_module import PlanetModule
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import argparse 


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


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    return parser.parse_args()


DATA_FOLDER = './data/'

if __name__ == '__main__':

    args = arg_parse()
    image_path = args.image_path  # Set image_path from command-line argument
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img, 224)

    model_path = args.model_path
    model = torch.jit.load(model_path, map_location='cpu')

    label_names = ['artisinal_mine', 
                    'selective_logging', 
                    'haze',
                    'slash_burn',
                    'clear',
                    'partly_cloudy',
                    'blow_down',
                    'bare_ground',
                    'water',
                    'habitation',
                    'road',
                    'conventional_mine',
                    'cultivation',
                    'primary',
                    'agriculture',
                    'blooming',
                    'cloudy', ]

    with torch.no_grad():
        probs = torch.sigmoid(model(img[None]))[0].detach().cpu().numpy()
    class_name2prob = {cls: prob for cls, prob in zip(label_names, probs)}
    print(class_name2prob)