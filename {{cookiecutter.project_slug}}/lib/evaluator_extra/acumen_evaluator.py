import os

class AcumenEvaluator(object):
    def __init__(self, args, model, logger = None):
        self.args = args
        self.model = model
        self.trainer = None
        self._logger = logger

        if 'output_dir' in self.args and self.args.output_dir is not None:
            os.makedirs(f'results/{self.args.output_dir}', exist_ok = True)

    @property
    def logger(self):
        if self.trainer is None:
            return self._logger

        return self.trainer.logger

    def evaluate(self):
        raise NotImplementedError

    def trainer_evaluate(self):
        raise NotImplementedError
