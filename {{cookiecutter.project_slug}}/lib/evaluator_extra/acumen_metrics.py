
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
            metric.log(logger)

class Metric(object):
    def __init__(self, name, value, log_direction = None):
        self.name = name
        self.value = value
        self.log_direction = log_direction

        if log_direction is None:
            # TODO add more metrics for best UX on the planet
            if ('f1' in self.name) or \
                    ('accuracy' in self.name) or\
                    ('precision' in self.name) or\
                    ('recall' in self.name):

                self.log_direction = 'up'

            elif ('loss' in self.name) or ('error' in self.name) or ('mae' in self.name) or ('mse' in self.name):
                self.log_direction = 'down'

            else:
                self.log_direction = None

    def log(self, logger):
        if self.log_direction == 'up':
            log_max = True
        elif self.log_direction == 'down':
            log_min = True
        elif self.log_direction == 'both':
            log_min = True
            log_max = True

        # TODO add log direction
        logger.log(f'{self.evaluator.display_name}#{self.name}', self.value, on_step = False, force_log = True)
