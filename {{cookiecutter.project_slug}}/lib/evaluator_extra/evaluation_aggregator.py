import torch
import numpy as np
from scipy.stats.mstats import gmean, hmean

from .acumen_metrics import MetricCollection, Metric
from .acumen_evaluator import AcumenEvaluator

class AcumenEvaluationAggregator(AcumenEvaluator):
    def __init__(self, args, model, evaluator_args = None, logger = None):
        super(AcumenEvaluationAggregator, self).__init__(args, model, evaluator_args, logger)

        self.is_aggregator = True

        # needs to contain
        # a list of metric names from other evaluators
        # an aggregation function name ("mean", "min", "max", "gmean", "hmean")
        # one or more directions ("up", "down", "instant")
        self.evaluator_args = evaluator_args

    @property
    def display_name(self):
        return 'aggregated'

    @torch.no_grad()
    def trainer_evaluate(self, step = None, metrics = None):
        if metrics is None:
            raise Exception('[AcumenEvaluationAggregator] No metrics to aggregate!')

        return self.evaluate(save = False, metrics = metrics)

    @torch.no_grad()
    def evaluate_and_log(self, global_step = -1, metrics = None):
        if metrics is None:
            raise Exception('[AcumenEvaluationAggregator] No metrics to aggregate!')

        metrics_of_interest = []
        for metric in metrics:
            if metric.log_name in self.evaluator_args.targets:
                metrics_of_interest.append(metric.value)

        if self.evaluator_args.agg_fn == 'mean':
            value = np.mean(metrics_of_interest)
        elif self.evaluator_args.agg_fn == 'min':
            value = np.min(metrics_of_interest)
        elif self.evaluator_args.agg_fn == 'max':
            value = np.max(metrics_of_interest)
        elif self.evaluator_args.agg_fn == 'gmean':
            value = gmean(metrics_of_interest)
        elif self.evaluator_args.agg_fn == 'hmean':
            value = hmean(metrics_of_interest)
        else:
            raise Exception(f'[AcumenEvaluationAggregator] agg_fn {self.evaluator_args.agg_fn} not understood!')

        out_metric = Metric(name = self.evaluator_args.agg_fn, value = value, evaluator = self, monotonicity = self.evaluator_args.direction)
        out_metric.log(self.logger)
