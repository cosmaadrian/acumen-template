import os
import glob

import torch
import numpy as np
from easydict import EasyDict
import json

# TODO add function to load checkpoint, optimizer and config.

def load_model(name = None, group = None, checkpoint_kind = 'best', checkpoint_path = None):
    if checkpoint_path is not None and (name is not None or group is not None):
        raise ValueError("checkpoint_path and name/group cannot be used together")

    if checkpoint_path is None:
        checkpoint_path = f'{group}:{name}/{checkpoint_kind}/'

    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/../{checkpoint_path}/*model.ckpt'
    print("::: Loading model from", checkpoint_path)
    checkpoints = glob.glob(checkpoint_path)

    try:
        state_dict = torch.load(checkpoints[-1])
    except Exception as e:
        print("No checkpoints found: ", checkpoint_path)
        raise e

    return state_dict

def load_config(name = None, group = None, checkpoint_kind = 'best', checkpoint_path = None):
    if checkpoint_path is not None and (name is not None or group is not None):
        raise ValueError("checkpoint_path and name/group cannot be used together")

    if checkpoint_path is None:
        checkpoint_path = f'{group}:{name}/{checkpoint_kind}/'

    config_path = f'{os.path.abspath(os.path.dirname(__file__))}/../{checkpoint_path}/*config.json'
    print("::: Loading config from", config_path)
    configs = glob.glob(config_path)

    try:
        with open(configs[-1], 'r') as f:
            config = EasyDict(json.load(f))
    except Exception as e:
        print("No configs found: ", config_path)
        raise e

    return config
