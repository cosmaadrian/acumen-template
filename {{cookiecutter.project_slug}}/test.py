from lib.arg_utils import define_args
import pprint

from lib import MultiHead
from lib import NotALightningTrainer
from lib import nomenclature

args = define_args()

train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)
architecture = nomenclature.MODELS[args.model](args)
model = nomenclature.TRAINERS[args.trainer](args, architecture)

trainer = NotALightningTrainer(args = args)

trainer.fit(
    model,
    train_dataloader,
    evaluators = [] 
)

print(model)
