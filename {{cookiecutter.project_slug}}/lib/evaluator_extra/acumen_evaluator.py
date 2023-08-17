import os
import torch

class AcumenEvaluator(object):
    def __init__(self, args, model, evaluator_args = None, logger = None):
        self.args = args
        self.model = model
        self.evaluator_args = evaluator_args
        self.trainer = None
        self._logger = logger

        if 'output_dir' in self.args and self.args.output_dir is not None:
            os.makedirs(f'results/{self.args.output_dir}', exist_ok = True)

    @property
    def logger(self):
        if self.trainer is None:
            return self._logger

        return self.trainer.logger

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    @torch.no_grad()
    def trainer_evaluate(self):
        raise NotImplementedError
