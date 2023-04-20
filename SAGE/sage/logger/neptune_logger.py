"""
Neptune logger class

Copyright (c) 2022 Hocheol Lim.
"""

from typing import List, Union

import neptune

from .abstract_logger import AbstractLogger


class NeptuneLogger(AbstractLogger):
    def __init__(self, args, tags):
        neptune.init(project_qualified_name=args.project_qualified_name)
        experiment = neptune.create_experiment(name="sage", params=vars(args))
        neptune.append_tag(tags)

    def log_metric(self, name: str, value: Union[int, float]):
        neptune.log_metric(name, value)

    def log_text(self, name: str, text: str):
        neptune.log_text(name, text)

    def log_values(self, name: str, values: List[float]):
        for idx in range(len(values)):
            neptune.log_metric("{}_{}".format(name, idx), values[idx])
