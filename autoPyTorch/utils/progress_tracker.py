# -*- encoding: utf-8 -*-
from typing import Dict
from abc import ABC, abstractmethod

import time
import torch

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
        self.isTracking = False
        self.total_time = total_time

    def set_time_left(self, time_seconds: float):
        self.total_time = time_seconds

    def start(self):
        """
        Start the tracking process
        """
        self.start_time = time.time()
        self.last_time_updated = self.start_time
        self.isTracking = True
        self.on_start()

    def on_start(self):
        pass

    def stop(self):
        """
        Stop the tracking process
        """
        self.stop_time = time.time()
        self.isTracking = False
        self.on_stop()

    def on_stop(self):
        pass

    def __call__(
        self,
        smbo: 'SMBO',
        result: RunValue,
        time_left: float,
        run_info: RunInfo
    ) -> None:

        if not self.isTracking:
            return
        cur_time = time.time()

        total_time_passed = cur_time - self.start_time
        total_time_left = self.total_time - total_time_passed

        time_passed = cur_time - self.last_time_updated
        self.last_time_updated = cur_time

        metrics = None
        if "tracked_metrics" in result.additional_info:
            metrics = result.additional_info["tracked_metrics"]

        model_name=None
        if "network_backbone:__choice__" in run_info.config:
            model_name = run_info.config["network_backbone:__choice__"]
        metrics = metrics if metrics else {}
        cost = result.cost
        self.report_progress(time_passed=time_passed, metrics=metrics, cost=cost, model=model_name)

    @abstractmethod
    def report_progress(
        self,
        time_passed: float,
        metrics: dict,
        model: str,
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

class TrainingEpochTracker(ABC):

    def __init__(self):
        self.total_steps = None
        self.isTracking = False

    def start(self, total_epoch_steps: int):
        self.start_time = time.time()
        self.isTracking = True
        self.total_steps = total_epoch_steps
        self.on_start(total_epoch_steps)

    def on_start(self, total_epoch_steps: int):
        pass

    def stop(self):
        self.stop_time = time.time()
        self.isTracking = False
        self.on_stop()

    def on_stop(self):
        pass

    def __call__(
        self,
        loss: float,
        result: torch.Tensor
    ) -> None:
        self.report_step_progress(loss, result)

    @abstractmethod
    def report_step_progress(
        self,
        loss: float,
        result: torch.Tensor
    ) -> None:
        """Callback that reports on step progress

        Args:
            loss (float):
                loss of the current epoch step
            result (torch.Tensor):
                result of the current training step
        """
        raise NotImplementedError("Function called on ProgressTracker, this can only be called by "
                                  "specific progress tracker")
