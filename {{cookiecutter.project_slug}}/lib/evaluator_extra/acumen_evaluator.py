import os
import torch
from .acumen_metrics import MetricCollection, Metric
import inflection


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
    def display_name(self):
        return inflection.underscore(self.__class__.__name__)

    @property
    def logger(self):
        if self.trainer is None:
            return self._logger

        return self.trainer.logger

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    @torch.no_grad()
    def trainer_evaluate(self, global_step = -1):
        raise NotImplementedError

    def evaluate_and_log(self, global_step = -1):
        outputs = self.trainer_evaluate(global_step)

        metric_collection = MetricCollection(evaluator = self)

        for output in outputs:
            if isinstance(output, Metric):
                metric_collection.append(output)
            else:
                key, value = output
                metric_collection.append(Metric(name = key, value = value))

        metric_collection.log(self._logger)
