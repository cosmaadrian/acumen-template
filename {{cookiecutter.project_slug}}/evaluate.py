import yaml
import pprint
from easydict import EasyDict

from lib import nomenclature
from lib import device

from lib.arg_utils import define_args
from lib.utils import load_model, load_config
from lib.loggers import NoLogger
from lib.forge import VersionCommand

VersionCommand().run()

args = define_args(
    require_config_file = False,
    extra_args = [
        ('--eval_batch_size', {'default': None, 'type': str, 'required': False})
        ('--eval_config', {'default': '', 'type': str, 'required': True}),
        ('--output_dir', {'default': '', 'type': str, 'required': True}),
        ('--checkpoint_kind', {'default': 'best', 'type': str, 'required': False})
    ])

with open(args.eval_config, 'rt') as f:
    eval_cfg = EasyDict(yaml.load(f, Loader = yaml.FullLoader))

model_config = load_config(args)

if args.eval_batch_size is None:
    args.eval_batch_size = model_config['eval_batch_size']

args = EasyDict({**model_config, **args})

architecture = nomenclature.MODELS[args.model](args)

state_dict = load_model(args)
state_dict = {
    key.replace('module.', ''): value
    for key, value in state_dict.items()
}

architecture.load_state_dict(state_dict)
architecture.eval()
architecture.train(False)
architecture.to(device)

evaluators = [
    nomenclature.EVALUATORS[evaluator_args.name](args, architecture, evaluator_args.args, logger = NoLogger())
    for evaluator_args in eval_cfg['evaluators']
]

for evaluator in evaluators:
    results = evaluator.evaluate(save = True)
    print(evaluator.__class__.__name__)
    pprint.pprint(results)
