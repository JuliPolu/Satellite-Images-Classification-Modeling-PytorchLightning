import typing as tp
import numpy as np
import torch
from src.lightning_module import PlanetModule
import argparse 


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    return parser.parse_args()

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, classes, size, threshold):
        super().__init__()
        self.model = model
        self.classes = classes
        self.size = size
        self.threshold = threshold
    
    
    def forward(self, x):
        return self.model(x)
    

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


if __name__ == '__main__':

    args = arg_parse()
    checkpoint_name = args.checkpoint  # Set checkpoint_name from command-line argument

    # model = PlanetModule.load_from_checkpoint(checkpoint_name, map_location=torch.device('cpu'))
    core_model = PlanetModule.load_from_checkpoint(checkpoint_name, map_location=torch.device('cpu'))._model

    model_wrapper = ModelWrapper(core_model, classes=label_names, size=224, threshold=0.2)

    dummy_input = torch.randn(1, 3, 224, 224)
    traced_scripted_model = torch.jit.script(model_wrapper, dummy_input)
    torch.jit.save(traced_scripted_model, "models/jit_model/final_model2.pt")