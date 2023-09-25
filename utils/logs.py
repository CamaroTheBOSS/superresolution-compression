import logging
import os.path
import time
from collections import defaultdict, deque
from datetime import timedelta, datetime
from typing import Any

import torch
import wandb


def log_to_wandb(loss_dict: dict, metrics_dict: dict) -> None:
    wandb.log({"loss_" + key: value for key, value in loss_dict.items()})
    wandb.log({"metric_" + key: value for key, value in metrics_dict.items()})


class FileLogger:
    def __init__(self, log_directory: str, filename: str = "train.log") -> None:
        logging.basicConfig(format='%(message)s', filename=f"{log_directory}/train.log", filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.log_dir = f"{log_directory}/{filename}"
        self.init_log_file()

    def init_log_file(self) -> None:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            with open(self.log_dir, 'w') as f:
                self.logger.debug("")

    def log(self, lines) -> None:
        if isinstance(lines, str):
            lines = [lines]
        with open(self.log_dir, 'a+') as f:
            f.writelines('\n')
            for line in lines:
                self.logger.debug(line)
            self.logger.debug('\n' * 2)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.force_save = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / (self.count + 0.00000001)

    @property
    def max(self):
        if len(self.deque):
            return max(self.deque)
        else:
            return 0

    @property
    def value(self):
        if len(self.deque):
            return self.deque[-1]
        else:
            return 0

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger:
    def __init__(self, delimiter="\t") -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.metrics = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.date_format = "%d-%m-%Y %H:%M:%S"

    def add_metric(self, name, metric) -> None:
        self.metrics[name] = metric

    def update(self, metrics: dict, prefix="") -> None:
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.metrics[prefix + key].update(value)

    def log(self, content, *args) -> None:
        for arg in args:
            content += str(arg)
        self.logger.info(content)

    def __getattr__(self, attr) -> SmoothedValue:
        if attr in self.metrics:
            return self.metrics[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.metrics.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def __call__(self, iterable, header=""):
        idx = 0
        start_time = time.time()
        iteration_time = SmoothedValue(fmt="{avg:.4f}")
        current_index = ":" + str(len(str(len(iterable)))) + "d"

        log_data = ["[{date}]", header, "[{0" + current_index + "}/{1}]", "{eta}", "{iteration_time}/it", "{metrics}",
                    "GPU memory usage: {memory:.0f}MB" if torch.cuda.is_available() else "{memory}"]
        log_msg = self.delimiter.join(log_data)

        for obj in iterable:
            yield obj
            iteration_time.update(time.time() - start_time)
            eta_seconds = iteration_time.global_avg * (len(iterable) - 1)
            eta_string = str(timedelta(seconds=int(eta_seconds)))
            memory = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else "NAN"
            self.log(log_msg.format(
                idx,
                len(iterable),
                date=datetime.now().strftime(self.date_format),
                eta=eta_string,
                metrics=str(self),
                iteration_time=str(iteration_time),
                memory=memory, ))
            start_time = time.time()
            idx += 1
