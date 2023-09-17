INCREASING_METRICS = ['accuracy', 'f1', 'precision', 'recall']
DECREASING_METRICS = ['loss', 'error', 'mae', 'mse']

class MetricCollection(object):
    def __init__(self, evaluator, metrics = None):
        self.metrics = metrics
        self.evaluator = evaluator

        if metrics is None:
            self.metrics = []
    def append(self, metric):
        self.metrics.append(metric)

    def log(self, logger):
        for metric in self.metrics:
            # this sucks, but oh well
            metric.log(logger, evaluator = self.evaluator)

class Metric(object):
    def __init__(self, name, value, monotonicity = None):
        self.name = name
        self.value = value
        self.monotonicity = monotonicity

        if monotonicity is None:
            if any([metric in self.name for metric in INCREASING_METRICS]):
                self.monotonicity = ['instant', 'up']

            elif any([metric in self.name for metric in DECREASING_METRICS]):
                self.monotonicity = ['instant', 'down']

            else:
                self.monotonicity = ['instant']

    def log(self, logger, evaluator):
        log_min = False
        log_max = False
        log_instant = False

        if 'up' in self.monotonicity:
            log_max = True

        if 'down' in self.monotonicity:
            log_min = True

        if 'instant' in self.monotonicity:
            log_instant = True

        logger.log(
            f'{evaluator.display_name}#{self.name}',
            self.value,
            on_step = False,
            force_log = True,
            log_max = log_max,
            log_min = log_min,
            log_instant = log_instant
        )
