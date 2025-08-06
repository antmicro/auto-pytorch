# -*- encoding: utf-8 -*-
from typing import Dict
from abc import ABC, abstractmethod

import time

from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue

class TrainingProgressTracker(IncorporateRunResultCallback, ABC):
    """
    Used to track progress in Optimization step

    Args:
        total_time (float):
            total time that is predicted for tracked task
    """

    def __init__(
        self,
        total_time: float,
    ):
        self.total_time = total_time

    def set_time_left(self, time_seconds: float):
        self.total_time = time_seconds

    def start(self):
        """
        Start the tracking process
        """
        self.start_time = time.time()
    def __call__(
        self,
        smbo: 'SMBO',
        result: RunValue,
        time_left: float,
        run_info: RunInfo
    ) -> None:

        total_time_passed = cur_time - self.start_time
        total_time_left = self.total_time - total_time_passed
        metrics = None
        if "tracked_metrics" in result.additional_info:
            metrics = result.additional_info["tracked_metrics"]

        metrics = metrics if metrics else {}
        cost=result.cost
        self.report_progress(time_passed=result.time, metrics=metrics, cost=cost, model=model_name)

    @abstractmethod
    def report_progress(
        self,
        time_passed: float,
        metrics: dict,
        cost
    ) -> None:
        """Callback that reports on assigned tasks' progress

        Args:
            time_passed (float):
                Time that has passed since last invocation of report_progress in seconds
            metrics (dict):
                Dictionary containing key-value pairs of metric_name-metric_value
            model (str):
                Model_backbone that is being reported on
            cost (float):
                Cost that is used to evaluate this model, the lower the better
        """

        raise NotImplementedError("Function called on ProgressTracker, this can only be called by "
                                  "specific progress tracker")
