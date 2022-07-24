import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


from models import *
MODELS = {

}

from datasets import *
DATASETS = {

}

from trainers import *
TRAINER = {

}


from evaluators import *
EVALUATORS = {

}
