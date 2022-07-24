import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
from torch.utils.data import DataLoader

import wandb

import callbacks
from lib import NotALightningTrainer
from loggers import WandbLogger

import nomenclature
from lib.arg_utils import define_args

args = define_args()
wandb.init(project = '{{cookiecutter.project_slug}}', group = args.group)
wandb.config.update(vars(args))

dataset = nomenclature.DATASETS[args.dataset]
train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)

architecture = nomenclature.MODELS[args.model](args)
model = nomenclature.TRAINER[args.trainer](args, architecture)

wandb_logger = WandbLogger()

checkpoint_callback = callbacks.ModelCheckpoint(
    monitor = f'{monitor_quantity}',
    dirpath = f'checkpoints/{args.group}:{args.name}',
    save_weights_only = True,
    direction='up',
    filename=f'epoch={% raw %}{{epoch}}{% endraw %}-val_acc={% raw %}{{{monitor_quantity}:.4f}}{% endraw %}.ckpt',
)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer = model.configure_optimizers(lr = scheduler_args.base_lr),
    cycle_momentum = False,
    base_lr = scheduler_args.base_lr,
    mode = scheduler_args.mode.,
    step_size_up = len(train_dataloader) * scheduler_args.step_size_up, # per epoch
    step_size_down = len(train_dataloader) * scheduler_args.step_size_down, # per epoch
    max_lr = scheduler_args.max_lr
)

lr_callback = callbacks.LambdaCallback(
    on_batch_end = lambda: scheduler.step()
)

lr_logger = callbacks.LambdaCallback(
    on_batch_end = lambda: wandb_logger.log('lr', lr_callback.get_last_lr()[0])
)

trainer = NotALightningTrainer(
    args = args,
    callbacks = [
        checkpoint_callback,
        lr_callback,
        lr_logger
    ],
    logger=wandb_logger,
)

torch.backends.cudnn.benchmark = True
trainer.fit(
    model,
    train_dataloader,
    evaluators = evaluators
)
