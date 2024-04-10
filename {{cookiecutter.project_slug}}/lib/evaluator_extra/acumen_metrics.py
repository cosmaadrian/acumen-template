INCREASING_METRICS = ['accuracy', 'f1', 'precision', 'recall']
DECREASING_METRICS = ['loss', 'error', 'mae', 'mse']

class MetricCollection(object):
    def __init__(self, metrics = None):
        self.metrics = metrics
        self.index = 0

        if metrics is None:
            self.metrics = []

    def append(self, metric):
        self.metrics.append(metric)

    def extend(self, metric_collection):
        for m in metric_collection:
            self.metrics.append(m)

    def log(self, logger):
        for metric in self.metrics:
            metric.log(logger)

    def __iter__(self):
        return self.metrics.__iter__()

class Metric(object):
    def __init__(self, name, value, evaluator = None, monotonicity = None):
        self.name = name
        self.value = value
        self.monotonicity = monotonicity
        self.evaluator = evaluator

        self.log_name =  f'{self.evaluator.display_name}#{self.name}'

        if monotonicity is None:
            if any([metric in self.name for metric in INCREASING_METRICS]):
                self.monotonicity = ['instant', 'up']

            elif any([metric in self.name for metric in DECREASING_METRICS]):
                self.monotonicity = ['instant', 'down']

            else:
                self.monotonicity = ['instant']

    def log(self, logger):
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
            self.log_name,
            self.value,
            on_step = False,
            force_log = True,
            log_max = log_max,
            log_min = log_min,
            log_instant = log_instant
        )
