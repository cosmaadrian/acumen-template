import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from ._nomenclature import NOMENCLATURE as nomenclature
from .trainer import NotALightningTrainer

from .model_extra import ModelOutput, ClassificationOutput, MultiHead
