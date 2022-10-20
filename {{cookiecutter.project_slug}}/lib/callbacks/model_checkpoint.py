import os
import torch

from .callback import Callback


class ModelCheckpoint(Callback):

    def __init__(self,
            name = "ModelCheckpoint",
            monitor = 'val_loss',
            direction = 'down',
            dirpath = 'checkpoints/',
            save_weights_only = False,
            filename="checkpoint",
            save_best_only = True,
            start_counting_at = 0
        ):

        self.start_counting_at = start_counting_at
        self.trainer = None
        self.monitor = monitor
        self.name = name
        self.direction = direction
        self.dirpath = dirpath
        self.save_weights_only = save_weights_only
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

        if self.save_weights_only:
            torch.save(self.trainer.model_hook.state_dict(), path)
        else:
            torch.save(self.trainer.model_hook, path)
