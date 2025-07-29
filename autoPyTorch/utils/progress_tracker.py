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
        self.report_progress(total_time_passed=total_time_passed, total_time_left=total_time_left)
    @abstractmethod
    def report_progress(
        self,
        total_time_passed: float,
        total_time_left: float
        # logger: PicklableClientLogger
    ) -> None:
        """Callback that reports on assigned tasks' progress

        Args:
            total_time_passed (float):
                total time that this task was already running
            total_time_left (float):
                total time that this task has still available
        """
        raise NotImplementedError("Function called on ProgressTracker, this can only be called by "
                                  "specific progress tracker")
