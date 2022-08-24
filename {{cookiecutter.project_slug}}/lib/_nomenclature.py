import easydict

from .model_extra import CoralHead, ClassificationHead, CoralLoss, AcumenCrossEntropy
from .model_extra import MnistCNN
from .trainer_extra import *
from .dataset_extra import MnistDataset

NOMENCLATURE = easydict.EasyDict({
	'TRAINERS': {
		'auto': AutoTrainer,
	},

	'HEADS': {
		'coral': CoralHead,
		'classification': ClassificationHead,
	},

	'LOSSES': {
		'coral': CoralLoss,
        'xe': AcumenCrossEntropy,
	},

	'DATASETS': {
        'mnist': MnistDataset,
	},

	'MODELS': {
        'mnist-cnn': MnistCNN,
	},

	'EVALUATORS': {
        # TODO classification evaluator
	},
})



import nomenclature

# Merging with user stuff.

for actor_type in ['MODELS', 'TRAINERS', 'DATASETS', 'EVALUATORS']:
	if actor_type not in nomenclature.__dict__:
		continue

	for key, value in nomenclature.__dict__[actor_type].items():
		if key in NOMENCLATURE[actor_type]:
			raise Exception(f'::: {key} already defined for {actor_type}.')

		NOMENCLATURE[actor_type][key] = value

