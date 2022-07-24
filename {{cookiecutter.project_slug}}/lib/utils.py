import os
import glob

import torch


def load_model(args):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/checkpoints/{args.group}:{args.name}/*.ckpt'
    print("::: Loading model from", checkpoint_path)
    checkpoints = glob.glob(checkpoint_path)

    try:
        state_dict = torch.load(checkpoints[-1])
    except Exception as e:
        print("No checkpoints found: ", checkpoint_path)
        raise e

    return state_dict

def load_model_by_dir(name):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/{name}/*.ckpt'
    print('::: Loading model from:', checkpoint_path)
    checkpoints = glob.glob(checkpoint_path)
    return torch.load(checkpoints[-1])

def load_model_by_name(name):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/checkpoints/{name}/*.ckpt'
    checkpoints = glob.glob(checkpoint_path)
    return torch.load(checkpoints[-1])
