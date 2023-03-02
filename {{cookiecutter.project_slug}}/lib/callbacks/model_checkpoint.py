import os
import torch
import json

from .callback import Callback


class ModelCheckpoint(Callback):

    def __init__(self,
            args,
            name = "ModelCheckpoint",
            monitor = 'val_loss',
            direction = 'down',
            dirpath = 'checkpoints/',
            filename="checkpoint",
            save_best_only = True,
            start_counting_at = 0
        ):

        self.args = args
        self.start_counting_at = start_counting_at
        self.trainer = None
        self.monitor = monitor
        self.name = name
        self.direction = direction
        self.dirpath = dirpath
        self.filename = filename
        self.save_best_only = save_best_only

        self.previous_best = None
        self.previous_best_path = None


    def on_epoch_end(self):
        if self.trainer.epoch < self.start_counting_at:
            return

        trainer_quantity = self.trainer.logger.metrics[self.monitor]

        if self.previous_best is not None and self.save_best_only:
            if self.direction == 'down':
                if self.previous_best <= trainer_quantity:
                    print(f"No improvement. Current: {trainer_quantity} - Previous {self.previous_best}")
                    return
            else:
                if self.previous_best >= trainer_quantity:
                    print(f"No improvement. Current: {trainer_quantity} - Previous {self.previous_best}")
                    return

        if self.previous_best_path is not None:
            os.unlink(self.previous_best_path)

        path = os.path.join(self.dirpath, self.filename.format(
            **{'epoch': self.trainer.epoch, self.monitor: trainer_quantity}
        ))

        print(f"[{self.name}] Saving model to: {path}")

        os.makedirs(self.dirpath, exist_ok = True)

        self.previous_best = trainer_quantity
        self.previous_best_path = path

        config_path = os.path.join(self.dirpath, 'config.json')

        if not os.path.exists(config_path):
            with open(config_path, 'wt') as f:
                json.dump(self.args, f, indent = 4)

        torch.save({
                'model_state_dict': self.trainer.model_hook.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'epoch': self.trainer.epoch,
            },
            path
        )
