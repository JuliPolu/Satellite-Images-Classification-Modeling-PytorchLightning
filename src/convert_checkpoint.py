import typing as tp
import numpy as np
import pandas as pd
import torch
# import onnx
# import onnxruntime as ort
from src.lightning_module import PlanetModule
import argparse 


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()

    checkpoint_name = args.checkpoint  # Set checkpoint_name from command-line argument

    model = PlanetModule.load_from_checkpoint(checkpoint_name, map_location=torch.device('cpu'))

    scripted_model = model.to_torchscript()
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_scripted_model = torch.jit.trace(scripted_model, dummy_input)
    torch.jit.save(traced_scripted_model, "models/jit_model/final_model.pt")







