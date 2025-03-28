# -*- encoding: utf-8 -*-
import glob
import gzip
import logging
import logging.handlers
import math
import multiprocessing
import numbers
import os
import pickle
import re
import shutil
import time
import traceback
import zlib
from typing import Dict, List, Optional, Set, Tuple, Union

import dask.distributed

import numpy as np

import pandas as pd

import pynisher

from sklearn.utils.validation import check_random_state

from smac.callbacks import IncorporateRunResultCallback
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import BINARY
from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.ensemble.ensemble_selection import EnsembleSelection
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss, calculate_score
from autoPyTorch.utils.logging_ import get_named_client_logger
from autoPyTorch.utils.parallel import preload_modules

Y_ENSEMBLE = 0
Y_TEST = 1

MODEL_FN_RE = r'_([0-9]*)_([0-9]*)_([0-9]+\.*[0-9]*)\.npy'


class EnsembleBuilderManager(IncorporateRunResultCallback):
    def __init__(
        self,
        start_time: float,
        time_left_for_ensembles: float,
        backend: Backend,
        dataset_name: str,
        task_type: int,
        output_type: int,
        metrics: List[autoPyTorchMetric],
        opt_metric: str,
        ensemble_size: int,
        ensemble_nbest: int,
        max_models_on_disc: Union[float, int],
        seed: int,
        precision: int,
        max_iterations: Optional[int],
        read_at_most: int,
        ensemble_memory_limit: Optional[int],
        random_state: int,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        pynisher_context: str = 'fork',
        metrics_kwargs: Dict = {},
    ):
        """ SMAC callback to handle ensemble building
        Args:
            start_time: int
                the time when this job was started, to account for any latency in job allocation
            time_left_for_ensemble: int
                How much time is left for the task. Job should finish within this allocated time
            backend: util.backend.Backend
                backend to write and read files
            dataset_name: str
                name of dataset
            task_type: int
                what type of output is expected. If Binary, we need to argmax the one hot encoding.
            metrics: List[autoPyTorchMetric],
                A set of metrics that will be used to get performance estimates
            opt_metric: str
                name of the optimization metrics
            ensemble_size: int
                maximal size of ensemble (passed to ensemble_selection)
            ensemble_nbest: int/float
                if int: consider only the n best prediction
                if float: consider only this fraction of the best models
                Both wrt to validation predictions
                If performance_range_threshold > 0, might return less models
            max_models_on_disc: Union[float, int]
                Defines the maximum number of models that are kept in the disc.
                If int, it must be greater or equal than 1, and dictates the max number of
                models to keep.
                If float, it will be interpreted as the max megabytes allowed of disc space. That
                is, if the number of ensemble candidates require more disc space than this float
                value, the worst models will be deleted to keep within this budget.
                Models and predictions of the worst-performing models will be deleted then.
                If None, the feature is disabled.
                It defines an upper bound on the models that can be used in the ensemble.
            seed: int
                random seed
            max_iterations: int
                maximal number of iterations to run this script
                (default None --> deactivated)
            precision (int): [16,32,64,128]
                precision of floats to read the predictions
            memory_limit: Optional[int]
                memory limit in mb. If ``None``, no memory limit is enforced.
            read_at_most: int
                read at most n new prediction files in each iteration
            logger_port: int
                port in where to publish a msg
            pynisher_context: str
                The multiprocessing context for pynisher. One of spawn/fork/forkserver.
            metrics_kwargs: Optional[Dict]
                Additioal information for computing metric

        Returns:
            List[Tuple[int, float, float, float]]:
                A list with the performance history of this ensemble, of the form
                [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
        """
        self.start_time = start_time
        self.time_left_for_ensembles = time_left_for_ensembles
        self.backend = backend
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.output_type = output_type
        self.metrics = metrics
        self.metrics_kwargs = metrics_kwargs
        self.opt_metric = opt_metric
        self.ensemble_size = ensemble_size
        self.ensemble_nbest = ensemble_nbest
        self.max_models_on_disc: Union[float, int] = max_models_on_disc
        self.seed = seed
        self.precision = precision
        self.max_iterations = max_iterations
        self.read_at_most = read_at_most
        self.ensemble_memory_limit = ensemble_memory_limit
        self.random_state = random_state
        self.logger_port = logger_port
        self.pynisher_context = pynisher_context

        # Store something similar to SMAC's runhistory
        self.history: List[Dict[str, float]] = []

        # We only submit new ensembles when there is not an active ensemble job
        self.futures: List[dask.Future] = []

        # The last criteria is the number of iterations
        self.iteration = 0

        # Keep track of when we started to know when we need to finish!
        self.start_time = time.time()

    def __call__(
        self,
        smbo: 'SMBO',
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> None:
        self.build_ensemble(smbo.tae_runner.client)

    def build_ensemble(
        self,
        dask_client: dask.distributed.Client,
        unit_test: bool = False
    ) -> None:

        # The second criteria is elapsed time
        elapsed_time = time.time() - self.start_time

        logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )

        # First test for termination conditions
        if self.time_left_for_ensembles < elapsed_time:
            logger.info(
                "Terminate ensemble building as not time is left (run for {}s)".format(
                    elapsed_time
                ),
            )
            return
        if self.max_iterations is not None and self.max_iterations <= self.iteration:
            logger.info(
                "Terminate ensemble building because of max iterations: {} of {}".format(
                    self.max_iterations,
                    self.iteration
                )
            )
            return

        if len(self.futures) != 0:
            if self.futures[0].done():
                result = self.futures.pop().result()
                if result:
                    ensemble_history, self.ensemble_nbest, _, _ = result
                    logger.debug("iteration={} @ elapsed_time={} has history={}".format(
                        self.iteration,
                        elapsed_time,
                        ensemble_history,
                    ))
                    self.history.extend(ensemble_history)

        # Only submit new jobs if the previous ensemble job finished
        if len(self.futures) == 0:

            # Add the result of the run
            # On the next while iteration, no references to
            # ensemble builder object, so it should be garbage collected to
            # save memory while waiting for resources
            # Also, notice how ensemble nbest is returned, so we don't waste
            # iterations testing if the deterministic predictions size can
            # be fitted in memory
            try:
                # Submit a Dask job from this job, to properly
                # see it in the dask diagnostic dashboard
                # Notice that the forked ensemble_builder_process will
                # wait for the below function to be done
                self.futures.append(dask_client.submit(
                    fit_and_return_ensemble,
                    backend=self.backend,
                    dataset_name=self.dataset_name,
                    task_type=self.task_type,
                    output_type=self.output_type,
                    metrics=self.metrics,
                    opt_metric=self.opt_metric,
                    ensemble_size=self.ensemble_size,
                    ensemble_nbest=self.ensemble_nbest,
                    max_models_on_disc=self.max_models_on_disc,
                    seed=self.seed,
                    precision=self.precision,
                    memory_limit=self.ensemble_memory_limit,
                    read_at_most=self.read_at_most,
                    random_state=self.seed,
                    end_at=self.start_time + self.time_left_for_ensembles,
                    iteration=self.iteration,
                    return_predictions=False,
                    priority=100,
                    pynisher_context=self.pynisher_context,
                    logger_port=self.logger_port,
                    unit_test=unit_test,
                    metric_kwargs=self.metrics_kwargs,
                ))

                logger.info(
                    "{}/{} Started Ensemble builder job at {} for iteration {}.".format(
                        # Log the client to make sure we
                        # remain connected to the scheduler
                        self.futures[0],
                        dask_client,
                        time.strftime("%Y.%m.%d-%H.%M.%S"),
                        self.iteration,
                    ),
                )
                self.iteration += 1
            except Exception as e:
                exception_traceback = traceback.format_exc()
                error_message = repr(e)
                logger.critical(exception_traceback)
                logger.critical(error_message)


def fit_and_return_ensemble(
    backend: Backend,
    dataset_name: str,
    task_type: int,
    output_type: int,
    metrics: List[autoPyTorchMetric],
    opt_metric: str,
    ensemble_size: int,
    ensemble_nbest: int,
    max_models_on_disc: Union[float, int],
    seed: int,
    precision: int,
    memory_limit: Optional[int],
    read_at_most: int,
    random_state: int,
    end_at: float,
    iteration: int,
    return_predictions: bool,
    pynisher_context: str,
    logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
    unit_test: bool = False,
    metric_kwargs: Dict = {}
) -> Tuple[
        List[Dict[str, float]],
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
]:
    """
    A short function to fit and create an ensemble. It is just a wrapper to easily send
    a request to dask to create an ensemble and clean the memory when finished
    Parameters
    ----------
        backend: util.backend.Backend
            backend to write and read files
        dataset_name: str
            name of dataset
        metrics: List[autoPyTorchMetric],
            A set of metrics that will be used to get performance estimates
        opt_metric:
            Name of the metric to optimize
        task_type: int
            type of output expected in the ground truth
        ensemble_size: int
            maximal size of ensemble (passed to ensemble.ensemble_selection)
        ensemble_nbest: int/float
            if int: consider only the n best prediction
            if float: consider only this fraction of the best models
            Both wrt to validation predictions
            If performance_range_threshold > 0, might return less models
        max_models_on_disc: int
           Defines the maximum number of models that are kept in the disc.
           If int, it must be greater or equal than 1, and dictates the max number of
           models to keep.
           If float, it will be interpreted as the max megabytes allowed of disc space. That
           is, if the number of ensemble candidates require more disc space than this float
           value, the worst models will be deleted to keep within this budget.
           Models and predictions of the worst-performing models will be deleted then.
           If None, the feature is disabled.
           It defines an upper bound on the models that can be used in the ensemble.
        seed: int
            random seed
        precision (int): [16,32,64,128]
            precision of floats to read the predictions
        memory_limit: Optional[int]
            memory limit in mb. If ``None``, no memory limit is enforced.
        read_at_most: int
            read at most n new prediction files in each iteration
        end_at: float
            At what time the job must finish. Needs to be the endtime and not the time left
            because we do not know when dask schedules the job.
        iteration: int
            The current iteration
        pynisher_context: str
            Context to use for multiprocessing, can be either fork, spawn or forkserver.
        logger_port: int
            The port where the logging server is listening to.
        unit_test: bool
            Turn on unit testing mode. This currently makes fit_ensemble raise a MemoryError.
            Having this is very bad coding style, but I did not find a way to make
            unittest.mock work through the pynisher with all spawn contexts. If you know a
            better solution, please let us know by opening an issue.
        metric_kwargs: Dict
            additional arguments for computing metrics, this is used for time series forecasting computation
    Returns
    -------
        List[Tuple[int, float, float, float]]
            A list with the performance history of this ensemble, of the form
            [[pandas_timestamp, train_performance, val_performance, test_performance], ...]
    """
    result = EnsembleBuilder(
        backend=backend,
        dataset_name=dataset_name,
        task_type=task_type,
        output_type=output_type,
        metrics=metrics,
        opt_metric=opt_metric,
        ensemble_size=ensemble_size,
        ensemble_nbest=ensemble_nbest,
        max_models_on_disc=max_models_on_disc,
        seed=seed,
        precision=precision,
        memory_limit=memory_limit,
        read_at_most=read_at_most,
        random_state=random_state,
        logger_port=logger_port,
        unit_test=unit_test,
        metric_kwargs=metric_kwargs,
    ).run(
        end_at=end_at,
        iteration=iteration,
        return_predictions=return_predictions,
        pynisher_context=pynisher_context,
    )
    return result


class EnsembleBuilder(object):
    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        task_type: int,
        output_type: int,
        metrics: List[autoPyTorchMetric],
        opt_metric: str,
        ensemble_size: int = 10,
        ensemble_nbest: int = 100,
        max_models_on_disc: Union[float, int] = 100,
        performance_range_threshold: float = 0,
        seed: int = 1,
        precision: int = 32,
        memory_limit: Optional[int] = 1024,
        read_at_most: int = 5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        unit_test: bool = False,
        metric_kwargs: Dict = {}
    ):
        """
            Constructor
            Parameters
            ----------
            backend: util.backend.Backend
                backend to write and read files
            dataset_name: str
                name of dataset
            task_type: int
                type of ML task
            metrics: List[autoPyTorchMetric],
                name of metric to score predictions
            opt_metric: str
                name of the metric to optimize
            ensemble_size: int
                maximal size of ensemble (passed to ensemble.ensemble_selection)
            ensemble_nbest: int/float
                if int: consider only the n best prediction
                if float: consider only this fraction of the best models
                Both wrt to validation predictions
                If performance_range_threshold > 0, might return less models
            max_models_on_disc: Union[float, int]
               Defines the maximum number of models that are kept in the disc.
               If int, it must be greater or equal than 1, and dictates the max number of
               models to keep.
               If float, it will be interpreted as the max megabytes allowed of disc space. That
               is, if the number of ensemble candidates require more disc space than this float
               value, the worst models will be deleted to keep within this budget.
               Models and predictions of the worst-performing models will be deleted then.
               If None, the feature is disabled.
               It defines an upper bound on the models that can be used in the ensemble.
            performance_range_threshold: float
                Keep only models that are better than:
                    dummy + (best - dummy)*performance_range_threshold
                E.g dummy=2, best=4, thresh=0.5 --> only consider models with score > 3
                Will at most return the minimum between ensemble_nbest models,
                and max_models_on_disc. Might return less
            seed: int
                random seed
            precision: [16,32,64,128]
                precision of floats to read the predictions
            memory_limit: Optional[int]
                memory limit in mb. If ``None``, no memory limit is enforced.
            read_at_most: int
                read at most n new prediction files in each iteration
            logger_port: int
                port that receives logging records
            unit_test: bool
                Turn on unit testing mode. This currently makes fit_ensemble raise a MemoryError.
                Having this is very bad coding style, but I did not find a way to make
                unittest.mock work through the pynisher with all spawn contexts. If you know a
                better solution, please let us know by opening an issue.
            metric_kwargs: Dict
                additional arguments for computing metrics, this is used for time series forecasting computation
        """

        super(EnsembleBuilder, self).__init__()

        self.backend = backend  # communication with filesystem
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.output_type = output_type
        self.metrics = metrics
        self.metric_kwargs = metric_kwargs
        self.opt_metric = opt_metric
        self.ensemble_size = ensemble_size
        self.performance_range_threshold = performance_range_threshold

        if isinstance(ensemble_nbest, numbers.Integral) and ensemble_nbest < 1:
            raise ValueError("Integer ensemble_nbest has to be larger 1: %s" %
                             ensemble_nbest)
        elif not isinstance(ensemble_nbest, numbers.Integral):
            if ensemble_nbest < 0 or ensemble_nbest > 1:
                raise ValueError(
                    "Float ensemble_nbest best has to be >= 0 and <= 1: %s" %
                    ensemble_nbest)

        self.ensemble_nbest = ensemble_nbest

        # max_models_on_disc can be a float, in such case we need to
        # remember the user specified Megabytes and translate this to
        # max number of ensemble models. max_resident_models keeps the
        # maximum number of models in disc
        if max_models_on_disc is not None and max_models_on_disc < 0:
            raise ValueError(
                "max_models_on_disc has to be a positive number or None"
            )
        self.max_models_on_disc = max_models_on_disc
        self.max_resident_models: Optional[int] = None

        self.seed = seed
        self.precision = precision
        self.memory_limit = memory_limit
        self.read_at_most = read_at_most
        self.random_state = check_random_state(random_state)
        self.unit_test = unit_test

        # Setup the logger
        self.logger_port = logger_port
        self.logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )

        if ensemble_nbest == 1:
            self.logger.debug("Behaviour depends on int/float: %s, %s (ensemble_nbest, type)" %
                              (ensemble_nbest, type(ensemble_nbest)))

        self.start_time = 0.0
        self.model_fn_re = re.compile(MODEL_FN_RE)

        self.last_hash = None  # hash of ensemble training data
        self.y_true_ensemble = None
        self.SAVE2DISC = True

        # already read prediction files
        # We read in back this object to give the ensemble the possibility to have memory
        # Every ensemble task is sent to dask as a function, that cannot take un-picklable
        # objects as attributes. For this reason, we dump to disk the stage of the past
        # ensemble iterations to kick-start the ensembling process
        # {"file name": {
        #    "ens_loss": float
        #    "mtime_ens": str,
        #    "mtime_test": str,
        #    "seed": int,
        #    "num_run": int,
        # }}
        self.read_losses = {}
        # {"file_name": {
        #    Y_ENSEMBLE: np.ndarray
        #    Y_TEST: np.ndarray
        #    }
        # }
        self.read_preds = {}

        # Depending on the dataset dimensions,
        # regenerating every iteration, the predictions
        # losses for self.read_preds
        # is too computationally expensive
        # As the ensemble builder is stateless
        # (every time the ensemble builder gets resources
        # from dask, it builds this object from scratch)
        # we save the state of this dictionary to memory
        # and read it if available
        self.ensemble_memory_file = os.path.join(
            self.backend.internals_directory,
            'ensemble_read_preds.pkl'
        )
        if os.path.exists(self.ensemble_memory_file):
            try:
                with (open(self.ensemble_memory_file, "rb")) as memory:
                    self.read_preds, self.last_hash = pickle.load(memory)
            except Exception as e:
                self.logger.warning(
                    "Could not load the previous iterations of ensemble_builder predictions."
                    "This might impact the quality of the run. Exception={} {}".format(
                        e,
                        traceback.format_exc(),
                    )
                )
        self.ensemble_loss_file = os.path.join(
            self.backend.internals_directory,
            'ensemble_read_losses.pkl'
        )
        if os.path.exists(self.ensemble_loss_file):
            try:
                with (open(self.ensemble_loss_file, "rb")) as memory:
                    self.read_losses = pickle.load(memory)
            except Exception as e:
                self.logger.warning(
                    "Could not load the previous iterations of ensemble_builder losses."
                    "This might impact the quality of the run. Exception={} {}".format(
                        e,
                        traceback.format_exc(),
                    )
                )

        # hidden feature which can be activated via an environment variable. This keeps all
        # models and predictions which have ever been a candidate. This is necessary to post-hoc
        # compute the whole ensemble building trajectory.
        self._has_been_candidate: Set[str] = set()

        self.validation_performance_ = np.inf

        # Track the ensemble performance
        self.y_test = None
        datamanager = self.backend.load_datamanager()
        if datamanager.test_tensors is not None:
            self.y_test = datamanager.test_tensors[1]
        del datamanager
        self.ensemble_history: List[Dict[str, float]] = []

    def run(
        self,
        iteration: int,
        pynisher_context: str,
        time_left: Optional[float] = None,
        end_at: Optional[float] = None,
        time_buffer: int = 5,
        return_predictions: bool = False,
    ) -> Tuple[
        List[Dict[str, float]],
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        This function is an interface to the main process and fundamentally calls main(), the
        later has the actual ensemble selection logic.

        The motivation towards this run() method is that it can be seen as a wrapper over the
        whole ensemble_builder.main() process so that pynisher can manage the memory/time limits.

        This is handy because this function reduces the number of members of the ensemble in case
        we run into memory issues. It does so in a halving fashion.

        Args:
            time_left (float):
                How much time is left for the ensemble builder process
            iteration (int):
                Which is the current iteration
            return_predictions (bool):
                Whether we want to return the predictions of the current model or not

        Returns:
            ensemble_history (Dict):
                A snapshot of both test and optimization performance. For debugging.
            ensemble_nbest (int):
                The user provides a direction on how many models to use in ensemble selection.
                This number can be reduced internally if the memory requirements force it.
            train_predictions (np.ndarray):
                The optimization prediction from the current ensemble.
            test_predictions (np.ndarray):
                The train prediction from the current ensemble.
        """

        if time_left is None and end_at is None:
            raise ValueError('Must provide either time_left or end_at.')
        elif time_left is not None and end_at is not None:
            raise ValueError('Cannot provide both time_left and end_at.')

        self.logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )

        process_start_time = time.time()
        while True:

            if time_left is not None:
                time_elapsed = time.time() - process_start_time
                time_left -= time_elapsed
            elif end_at is not None:
                current_time = time.time()
                if current_time > end_at:
                    break
                else:
                    time_left = end_at - current_time
            else:
                raise NotImplementedError()

            wall_time_in_s = int(time_left - time_buffer)
            if wall_time_in_s < 1:
                break
            context = multiprocessing.get_context(pynisher_context)
            preload_modules(context)

            safe_ensemble_script = pynisher.enforce_limits(
                wall_time_in_s=wall_time_in_s,
                mem_in_mb=self.memory_limit,
                logger=self.logger,
                context=context,
            )(self.main)
            safe_ensemble_script(time_left, iteration, return_predictions)
            if safe_ensemble_script.exit_status is pynisher.MemorylimitException:
                # if ensemble script died because of memory error,
                # reduce nbest to reduce memory consumption and try it again

                # ATTENTION: main will start from scratch; # all data structures are empty again
                try:
                    os.remove(self.ensemble_memory_file)
                except:  # noqa E722
                    pass

                if isinstance(self.ensemble_nbest, numbers.Integral) and self.ensemble_nbest <= 1:
                    if self.read_at_most == 1:
                        self.logger.error(
                            "Memory Exception -- Unable to further reduce the number of ensemble "
                            "members and can no further limit the number of ensemble members "
                            "loaded per iteration -- please restart autoPytorch with a higher "
                            "value for the argument `memory_limit` (current limit is %s MB). "
                            "The ensemble builder will keep running to delete files from disk in "
                            "case this was enabled.", self.memory_limit
                        )
                        self.ensemble_nbest = 0
                    else:
                        self.read_at_most = 1
                        self.logger.warning(
                            "Memory Exception -- Unable to further reduce the number of ensemble "
                            "members -- Now reducing the number of predictions per call to read "
                            "at most to 1."
                        )
                else:
                    if isinstance(self.ensemble_nbest, numbers.Integral):
                        self.ensemble_nbest = max(1, int(self.ensemble_nbest / 2))
                    else:
                        self.ensemble_nbest = int(self.ensemble_nbest / 2)
                    self.logger.warning("Memory Exception -- restart with "
                                        "less ensemble_nbest: %d" % self.ensemble_nbest)
                    return [], self.ensemble_nbest, None, None
            else:
                return safe_ensemble_script.result

        return [], self.ensemble_nbest, None, None

    def main(
        self, time_left: float, iteration: int, return_predictions: bool,
    ) -> Tuple[
        List[Dict[str, float]],
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        This is the main function of the ensemble builder process and can be considered
        a wrapper over the ensemble selection method implemented y EnsembleSelection class.

        This method is going to be called multiple times by the main process, to
        build and ensemble, in case the SMAC process produced new models and to provide
        anytime results.

        On this regard, this method mainly:
            1- select from all the individual models that smac created, the N-best candidates
               (this in the scenario that N > ensemble_nbest argument to this class). This is
               done based on a score calculated via the metrics argument.
            2- This pre-selected candidates are provided to the ensemble selection method
               and if a ensemble is found under the provided memory/time constraints, a new
               ensemble is proposed.
            3- Because this process will be called multiple times, it performs checks to make
               sure a new ensenmble is only proposed if new predictions are available, as well
               as making sure we do not run out of resources (like disk space)

        Args:
            time_left (float):
                How much time is left for the ensemble builder process
            iteration (int):
                Which is the current iteration
            return_predictions (bool):
                Whether we want to return the predictions of the current model or not

        Returns:
            ensemble_history (Dict):
                A snapshot of both test and optimization performance. For debugging.
            ensemble_nbest (int):
                The user provides a direction on how many models to use in ensemble selection.
                This number can be reduced internally if the memory requirements force it.
            train_predictions (np.ndarray):
                The optimization prediction from the current ensemble.
            test_predictions (np.ndarray):
                The train prediction from the current ensemble.
        """

        # Pynisher jobs inside dask 'forget'
        # the logger configuration. So we have to set it up
        # accordingly
        self.logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )

        self.start_time = time.time()
        train_pred, test_pred = None, None

        used_time = time.time() - self.start_time
        self.logger.debug(
            'Starting iteration %d, time left: %f',
            iteration,
            time_left - used_time,
        )

        # populates self.read_preds and self.read_losses
        if not self.compute_loss_per_model():
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None

        # Only the models with the n_best predictions are candidates
        # to be in the ensemble
        candidate_models = self.get_n_best_preds()
        if not candidate_models:  # no candidates yet
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None

        # populates predictions in self.read_preds
        # reduces selected models if file reading failed
        n_sel_test = self.get_test_preds(selected_keys=candidate_models)

        # If any of n_sel_* is not empty and overlaps with candidate_models,
        # then ensure candidate_models AND n_sel_test are sorted the same
        candidate_models_set = set(candidate_models)
        if candidate_models_set.intersection(n_sel_test):
            candidate_models = sorted(list(candidate_models_set.intersection(
                n_sel_test)))
            n_sel_test = candidate_models
        else:
            # This has to be the case
            n_sel_test = []

        if os.environ.get('ENSEMBLE_KEEP_ALL_CANDIDATES'):
            for candidate in candidate_models:
                self._has_been_candidate.add(candidate)

        # train ensemble
        ensemble = self.fit_ensemble(selected_keys=candidate_models)

        # Save the ensemble for later use in the main module!
        if ensemble is not None and self.SAVE2DISC:
            self.backend.save_ensemble(ensemble, iteration, self.seed)

        # Delete files of non-candidate models - can only be done after fitting the ensemble and
        # saving it to disc so we do not accidentally delete models in the previous ensemble
        if self.max_resident_models is not None:
            self._delete_excess_models(selected_keys=candidate_models)

        # Save the read losses status for the next iteration
        with open(self.ensemble_loss_file, "wb") as memory:
            pickle.dump(self.read_losses, memory)

        if ensemble is not None:
            train_pred = self.predict(set_="train",
                                      ensemble=ensemble,
                                      selected_keys=candidate_models,
                                      n_preds=len(candidate_models),
                                      index_run=iteration)
            # TODO if predictions fails, build the model again during the
            #  next iteration!
            test_pred = self.predict(set_="test",
                                     ensemble=ensemble,
                                     selected_keys=n_sel_test,
                                     n_preds=len(candidate_models),
                                     index_run=iteration)

            # Add a score to run history to see ensemble progress
            self._add_ensemble_trajectory(
                train_pred,
                test_pred
            )

        # The loaded predictions and the hash can only be saved after the ensemble has been
        # built, because the hash is computed during the construction of the ensemble
        with open(self.ensemble_memory_file, "wb") as memory:
            pickle.dump((self.read_preds, self.last_hash), memory)

        if return_predictions:
            return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
        else:
            return self.ensemble_history, self.ensemble_nbest, None, None

    def get_disk_consumption(self, pred_path: str) -> float:
        """
        gets the cost of a model being on disc
        """

        match = self.model_fn_re.search(pred_path)
        if not match:
            raise ValueError("Invalid path format %s" % pred_path)
        _seed = int(match.group(1))
        _num_run = int(match.group(2))
        _budget = float(match.group(3))

        stored_files_for_run = os.listdir(
            self.backend.get_numrun_directory(_seed, _num_run, _budget))
        stored_files_for_run = [
            os.path.join(self.backend.get_numrun_directory(_seed, _num_run, _budget), file_name)
            for file_name in stored_files_for_run]
        this_model_cost = sum([os.path.getsize(path) for path in stored_files_for_run])

        # get the megabytes
        return round(this_model_cost / math.pow(1024, 2), 2)

    def compute_loss_per_model(self) -> bool:
        """
            Compute the loss of the predictions on ensemble building data set;
            populates self.read_preds and self.read_losses
        """

        self.logger.debug("Read ensemble data set predictions")

        if self.y_true_ensemble is None:
            try:
                self.y_true_ensemble = self.backend.load_targets_ensemble()
            except FileNotFoundError:
                self.logger.debug(
                    "Could not find true targets on ensemble data set: %s",
                    traceback.format_exc(),
                )
                return False

        pred_path = os.path.join(
            glob.escape(self.backend.get_runs_directory()),
            '%d_*_*' % self.seed,
            'predictions_ensemble_%s_*_*.npy*' % self.seed,
        )
        y_ens_files = glob.glob(pred_path)
        y_ens_files = [y_ens_file for y_ens_file in y_ens_files
                       if y_ens_file.endswith('.npy') or y_ens_file.endswith('.npy.gz')]
        self.y_ens_files = y_ens_files
        # no validation predictions so far -- no files
        if len(self.y_ens_files) == 0:
            self.logger.debug("Found no prediction files on ensemble data set:"
                              " %s" % pred_path)
            return False

        # First sort files chronologically
        to_read = []
        for y_ens_fn in self.y_ens_files:
            match = self.model_fn_re.search(y_ens_fn)
            if match is None:
                raise ValueError(f"Could not interpret file {y_ens_fn} "
                                 "Something went wrong while scoring predictions")
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))

            to_read.append([y_ens_fn, match, _seed, _num_run, _budget])

        n_read_files = 0
        # Now read file wrt to num_run
        # Mypy assumes sorted returns an object because of the lambda. Can't get to recognize the list
        # as a returning list, so as a work-around we skip next line
        for y_ens_fn, match, _seed, _num_run, _budget in sorted(to_read, key=lambda x: x[3]):  # type: ignore
            if self.read_at_most and n_read_files >= self.read_at_most:
                # limit the number of files that will be read
                # to limit memory consumption
                break

            if not y_ens_fn.endswith(".npy") and not y_ens_fn.endswith(".npy.gz"):
                self.logger.info('Error loading file (not .npy or .npy.gz): %s', y_ens_fn)
                continue

            if not self.read_losses.get(y_ens_fn):
                self.read_losses[y_ens_fn] = {
                    "ens_loss": np.inf,
                    "mtime_ens": 0,
                    "mtime_test": 0,
                    "seed": _seed,
                    "num_run": _num_run,
                    "budget": _budget,
                    "disc_space_cost_mb": None,
                    # Lazy keys so far:
                    # 0 - not loaded
                    # 1 - loaded and in memory
                    # 2 - loaded but dropped again
                    # 3 - deleted from disk due to space constraints
                    "loaded": 0
                }
            if not self.read_preds.get(y_ens_fn):
                self.read_preds[y_ens_fn] = {
                    Y_ENSEMBLE: None,
                    Y_TEST: None,
                }

            if self.read_losses[y_ens_fn]["mtime_ens"] == os.path.getmtime(y_ens_fn):
                # same time stamp; nothing changed;
                continue

            # actually read the predictions and compute their respective loss
            try:
                y_ensemble = self._read_np_fn(y_ens_fn)
                losses = calculate_loss(
                    metrics=self.metrics,
                    target=self.y_true_ensemble,
                    prediction=y_ensemble,
                    task_type=self.task_type,
                    **self.metric_kwargs
                )

                if np.isfinite(self.read_losses[y_ens_fn]["ens_loss"]):
                    self.logger.debug(
                        'Changing ensemble loss for file %s from %f to %f '
                        'because file modification time changed? %f - %f',
                        y_ens_fn,
                        self.read_losses[y_ens_fn]["ens_loss"],
                        losses[self.opt_metric],
                        self.read_losses[y_ens_fn]["mtime_ens"],
                        os.path.getmtime(y_ens_fn),
                    )

                self.read_losses[y_ens_fn]["ens_loss"] = losses[self.opt_metric]

                # It is not needed to create the object here
                # To save memory, we just compute the loss.
                self.read_losses[y_ens_fn]["mtime_ens"] = os.path.getmtime(y_ens_fn)
                self.read_losses[y_ens_fn]["loaded"] = 2
                self.read_losses[y_ens_fn]["disc_space_cost_mb"] = self.get_disk_consumption(
                    y_ens_fn
                )

                n_read_files += 1

            except Exception:
                self.logger.warning(
                    'Error loading %s: %s',
                    y_ens_fn,
                    traceback.format_exc(),
                )
                self.read_losses[y_ens_fn]["ens_loss"] = np.inf

        self.logger.debug(
            'Done reading %d new prediction files. Loaded %d predictions in '
            'total.',
            n_read_files,
            np.sum([pred["loaded"] > 0 for pred in self.read_losses.values()])
        )
        return True

    def get_n_best_preds(self) -> List[str]:
        """
            get best n predictions (i.e., keys of self.read_losses)
            according to the loss on the "ensemble set"
            n: self.ensemble_nbest

            Side effects:
                ->Define the n-best models to use in ensemble
                ->Only the best models are loaded
                ->Any model that is not best is candidate to deletion
                  if max models in disc is exceeded.
        """

        sorted_keys = self._get_list_of_sorted_preds()

        # number of models available
        num_keys = len(sorted_keys)
        # remove all that are at most as good as random
        # note: dummy model must have run_id=1 (there is no run_id=0)
        dummy_losses = list(filter(lambda x: x[2] == 1, sorted_keys))
        # Leave this here for when we enable dummy classifier/scorer
        if len(dummy_losses) > 0:
            # number of dummy models
            num_dummy = len(dummy_losses)
            dummy_loss = dummy_losses[0]
            self.logger.debug("Use %f as dummy loss" % dummy_loss[1])
            sorted_keys = list(filter(lambda x: x[1] < dummy_loss[1], sorted_keys))

            # remove Dummy Classifier
            sorted_keys = list(filter(lambda x: x[2] > 1, sorted_keys))
            if len(sorted_keys) == 0:
                # no model left; try to use dummy loss (num_run==0)
                # log warning when there are other models but not better than dummy model
                if num_keys > num_dummy:
                    self.logger.warning("No models better than random - using Dummy Score!"
                                        "Number of models besides current dummy model: %d. "
                                        "Number of dummy models: %d",
                                        num_keys - 1,
                                        num_dummy)
                sorted_keys = [
                    (k, v["ens_loss"], v["num_run"]) for k, v in self.read_losses.items()
                    if v["seed"] == self.seed and v["num_run"] == 1
                ]
        # reload predictions if losses changed over time and a model is
        # considered to be in the top models again!
        if not isinstance(self.ensemble_nbest, numbers.Integral):
            # Transform to number of models to keep. Keep at least one
            keep_nbest = max(1, min(len(sorted_keys),
                                    int(len(sorted_keys) * self.ensemble_nbest)))
            self.logger.debug(
                "Library pruning: using only top %f percent of the models for ensemble "
                "(%d out of %d)",
                self.ensemble_nbest * 100, keep_nbest, len(sorted_keys)
            )
        else:
            # Keep only at most ensemble_nbest
            keep_nbest = min(self.ensemble_nbest, len(sorted_keys))
            self.logger.debug("Library Pruning: using for ensemble only "
                              " %d (out of %d) models" % (keep_nbest, len(sorted_keys)))

        # If max_models_on_disc is None, do nothing
        # One can only read at most max_models_on_disc models
        if self.max_models_on_disc is not None:
            if not isinstance(self.max_models_on_disc, numbers.Integral):
                consumption = [
                    [
                        v["ens_loss"],
                        v["disc_space_cost_mb"],
                    ] for v in self.read_losses.values() if v["disc_space_cost_mb"] is not None
                ]
                max_consumption = max(c[1] for c in consumption)

                # We are pessimistic with the consumption limit indicated by
                # max_models_on_disc by 1 model. Such model is assumed to spend
                # max_consumption megabytes
                if (sum(c[1] for c in consumption) + max_consumption) > self.max_models_on_disc:

                    # just leave the best -- smaller is better!
                    # This list is in descending order, to preserve the best models
                    sorted_cum_consumption = np.cumsum([
                        c[1] for c in list(sorted(consumption))
                    ]) + max_consumption
                    max_models = np.argmax(sorted_cum_consumption > self.max_models_on_disc)

                    # Make sure that at least 1 model survives
                    self.max_resident_models = max(1, max_models)
                    self.logger.warning(
                        "Limiting num of models via float max_models_on_disc={}"
                        " as accumulated={} worst={} num_models={}".format(
                            self.max_models_on_disc,
                            (sum(c[1] for c in consumption) + max_consumption),
                            max_consumption,
                            self.max_resident_models
                        )
                    )
                else:
                    self.max_resident_models = None
            else:
                self.max_resident_models = self.max_models_on_disc

        if self.max_resident_models is not None and keep_nbest > self.max_resident_models:
            self.logger.debug(
                "Restricting the number of models to %d instead of %d due to argument "
                "max_models_on_disc",
                self.max_resident_models, keep_nbest,
            )
            keep_nbest = self.max_resident_models

        # consider performance_range_threshold
        if self.performance_range_threshold > 0:
            best_loss = sorted_keys[0][1]
            worst_loss = dummy_loss[1]
            worst_loss -= (worst_loss - best_loss) * self.performance_range_threshold
            if sorted_keys[keep_nbest - 1][1] > worst_loss:
                # We can further reduce number of models
                # since worst model is worse than thresh
                for i in range(0, keep_nbest):
                    # Look at most at keep_nbest models,
                    # but always keep at least one model
                    current_loss = sorted_keys[i][1]
                    if current_loss >= worst_loss:
                        self.logger.debug("Dynamic Performance range: "
                                          "Further reduce from %d to %d models",
                                          keep_nbest, max(1, i))
                        keep_nbest = max(1, i)
                        break
        ensemble_n_best = keep_nbest

        # reduce to keys
        reduced_sorted_keys = list(map(lambda x: x[0], sorted_keys))

        # remove loaded predictions for non-winning models
        for k in reduced_sorted_keys[ensemble_n_best:]:
            if k in self.read_preds:
                self.read_preds[k][Y_ENSEMBLE] = None
                self.read_preds[k][Y_TEST] = None
            if self.read_losses[k]['loaded'] == 1:
                self.logger.debug(
                    'Dropping model %s (%d,%d) with loss %f.',
                    k,
                    self.read_losses[k]['seed'],
                    self.read_losses[k]['num_run'],
                    self.read_losses[k]['ens_loss'],
                )
                self.read_losses[k]['loaded'] = 2

        # Load the predictions for the winning
        for k in reduced_sorted_keys[:ensemble_n_best]:
            if (
                (
                    k not in self.read_preds or self.read_preds[k][Y_ENSEMBLE] is None
                )
                and self.read_losses[k]['loaded'] != 3
            ):
                self.read_preds[k][Y_ENSEMBLE] = self._read_np_fn(k)
                # No need to load test here because they are loaded
                #  only if the model ends up in the ensemble
                self.read_losses[k]['loaded'] = 1

        # return best scored keys of self.read_losses
        return reduced_sorted_keys[:ensemble_n_best]

    def get_test_preds(self, selected_keys: List[str]) -> List[str]:
        """
        test predictions from disc
        and store them in self.read_preds
        Parameters
        ---------
        selected_keys: list
            list of selected keys of self.read_preds
        Return
        ------
        success_keys:
            all keys in selected keys for which we could read the valid and
            test predictions
        """
        success_keys_test = []

        for k in selected_keys:
            test_fn = glob.glob(
                os.path.join(
                    glob.escape(self.backend.get_runs_directory()),
                    '%d_%d_%s' % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"],
                    ),
                    'predictions_test_%d_%d_%s.npy*' % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"]
                    )
                )
            )
            test_fn = [tfn for tfn in test_fn if tfn.endswith('.npy') or tfn.endswith('.npy.gz')]

            if len(test_fn) == 0:
                # self.logger.debug("Not found test prediction file (although "
                #                   "ensemble predictions available):%s" %
                #                   test_fn)
                pass
            else:
                if (
                    self.read_losses[k]["mtime_test"] == os.path.getmtime(test_fn[0])
                    and k in self.read_preds
                    and self.read_preds[k][Y_TEST] is not None
                ):
                    success_keys_test.append(k)
                    continue
                try:
                    y_test = self._read_np_fn(test_fn[0])
                    self.read_preds[k][Y_TEST] = y_test
                    success_keys_test.append(k)
                    self.read_losses[k]["mtime_test"] = os.path.getmtime(test_fn[0])
                except Exception:
                    self.logger.warning('Error loading %s: %s',
                                        test_fn, traceback.format_exc())

        return success_keys_test

    def fit_ensemble(self, selected_keys: List[str]) -> Optional[EnsembleSelection]:
        """
            fit ensemble

            Parameters
            ---------
            selected_keys: list
                list of selected keys of self.read_losses

            Returns
            -------
            ensemble: EnsembleSelection
                trained Ensemble
        """

        if self.unit_test:
            raise MemoryError()

        predictions_train = [self.read_preds[k][Y_ENSEMBLE] for k in selected_keys]
        include_num_runs = [
            (
                self.read_losses[k]["seed"],
                self.read_losses[k]["num_run"],
                self.read_losses[k]["budget"],
            )
            for k in selected_keys]

        # check hash if ensemble training data changed
        current_hash = "".join([
            str(zlib.adler32(predictions_train[i].data.tobytes()))
            for i in range(len(predictions_train))
        ])
        if self.last_hash == current_hash:
            self.logger.debug(
                "No new model predictions selected -- skip ensemble building "
                "-- current performance: %f",
                self.validation_performance_,
            )

            return None
        self.last_hash = current_hash

        opt_metric = [m for m in self.metrics if m.name == self.opt_metric][0]
        if not opt_metric:
            raise ValueError(f"Cannot optimize for {self.opt_metric} in {self.metrics} "
                             "as more than one unique optimization metric was found.")

        ensemble = EnsembleSelection(
            ensemble_size=self.ensemble_size,
            metric=opt_metric,
            random_state=self.random_state,
            task_type=self.task_type,
            metric_kwargs=self.metric_kwargs
        )

        try:
            self.logger.debug(
                "Fitting the ensemble on %d models.",
                len(predictions_train),
            )
            start_time = time.time()
            ensemble.fit(predictions_train, self.y_true_ensemble,
                         include_num_runs)
            end_time = time.time()
            self.logger.debug(
                "Fitting the ensemble took %.2f seconds.",
                end_time - start_time,
            )
            self.logger.info(str(ensemble))
            self.validation_performance_ = min(
                self.validation_performance_,
                ensemble.get_validation_performance(),
            )

        except ValueError:
            self.logger.error('Caught ValueError: %s', traceback.format_exc())
            return None
        except IndexError:
            self.logger.error('Caught IndexError: %s' + traceback.format_exc())
            return None
        finally:
            # Explicitly free memory
            del predictions_train

        return ensemble

    def predict(self, set_: str,
                ensemble: AbstractEnsemble,
                selected_keys: list,
                n_preds: int,
                index_run: int) -> np.ndarray:
        """
            save preditions on ensemble, validation and test data on disc
            Parameters
            ----------
            set_: ["test"]
                data split name
            ensemble: EnsembleSelection
                trained Ensemble
            selected_keys: list
                list of selected keys of self.read_losses
            n_preds: int
                number of prediction models used for ensemble building
                same number of predictions on valid and test are necessary
            index_run: int
                n-th time that ensemble predictions are written to disc
            Return
            ------
            y: np.ndarray
        """
        self.logger.debug("Predicting the %s set with the ensemble!", set_)

        if set_ == 'test':
            pred_set = Y_TEST
        else:
            pred_set = Y_ENSEMBLE
        predictions = [self.read_preds[k][pred_set] for k in selected_keys]

        if n_preds == len(predictions):
            y = ensemble.predict(predictions)
            if self.output_type == BINARY:
                y = y[:, 1 if y.shape[1] > 1 else 0]
            if self.SAVE2DISC:
                self.backend.save_predictions_as_txt(
                    predictions=y,
                    subset=set_,
                    idx=index_run,
                    prefix=self.dataset_name,
                    precision=8,
                )
            return y
        else:
            self.logger.info(
                "Found inconsistent number of predictions and models (%d vs "
                "%d) for subset %s",
                len(predictions),
                n_preds,
                set_,
            )
            return None

    def _add_ensemble_trajectory(self, train_pred: np.ndarray, test_pred: np.ndarray) -> None:
        """
        Records a snapshot of how the performance look at a given training
        time.
        Parameters
        ----------
        ensemble: EnsembleSelection
            The ensemble selection object to record
        test_pred: np.ndarray
            The predictions on the test set using ensemble
        """
        performance_stamp = {
            'Timestamp': pd.Timestamp.now(),
        }
        if self.output_type == BINARY:
            if len(train_pred.shape) == 1 or train_pred.shape[1] == 1:
                train_pred = np.vstack(
                    ((1 - train_pred).reshape((1, -1)), train_pred.reshape((1, -1)))
                ).transpose()
            if test_pred is not None and (len(test_pred.shape) == 1 or test_pred.shape[1] == 1):
                test_pred = np.vstack(
                    ((1 - test_pred).reshape((1, -1)), test_pred.reshape((1, -1)))
                ).transpose()

        train_scores = calculate_score(
            metrics=self.metrics,
            target=self.y_true_ensemble,
            prediction=train_pred,
            task_type=self.task_type,
            **self.metric_kwargs
        )
        performance_stamp.update({'train_' + str(key): val for key, val in train_scores.items()})
        if self.y_test is not None:
            test_scores = calculate_score(
                metrics=self.metrics,
                target=self.y_test,
                prediction=test_pred,
                task_type=self.task_type,
                **self.metric_kwargs
            )
            performance_stamp.update(
                {'test_' + str(key): val for key, val in test_scores.items()})

        self.ensemble_history.append(performance_stamp)

    def _get_list_of_sorted_preds(self) -> List[Tuple[str, float, int]]:
        """
            Returns a list of sorted predictions in descending performance order.
            (We are solving a minimization problem)
            Losses are taken from self.read_losses.

            Parameters
            ----------
            None

            Return
            ------
            sorted_keys:
                given a sequence of pairs of (loss[i], num_run[i]) = (l[i], n[i]),
                we will sort s.t. l[0] <= l[1] <= ... <= l[N] and for any pairs of
                i, j (i < j, l[i] = l[j]), the resulting sequence satisfies n[i] <= n[j]
        """
        # Sort by loss - smaller is better!
        sorted_keys = list(sorted(
            [
                (k, v["ens_loss"], v["num_run"])
                for k, v in self.read_losses.items()
            ],
            # Sort by loss as priority 1 and then by num_run on a ascending order
            # We want small num_run first
            key=lambda x: (x[1], x[2]),
        ))
        return sorted_keys

    def _delete_excess_models(self, selected_keys: List[str]) -> None:
        """
            Deletes models excess models on disc. self.max_models_on_disc
            defines the upper limit on how many models to keep.
            Any additional model with a worse loss than the top
            self.max_models_on_disc is deleted.
        """

        # Comply with mypy
        if self.max_resident_models is None:
            return

        # Obtain a list of sorted pred keys
        pre_sorted_keys = self._get_list_of_sorted_preds()
        sorted_keys = list(map(lambda x: x[0], pre_sorted_keys))

        if len(sorted_keys) <= self.max_resident_models:
            # Don't waste time if not enough models to delete
            return

        # The top self.max_resident_models models would be the candidates
        # Any other low performance model will be deleted
        # The list is in ascending order of score
        candidates = sorted_keys[:self.max_resident_models]

        # Loop through the files currently in the directory
        for pred_path in self.y_ens_files:

            # Do not delete candidates
            if pred_path in candidates:
                continue

            if pred_path in self._has_been_candidate:
                continue

            match = self.model_fn_re.search(pred_path)
            if match is None:
                raise ValueError("Could not interpret file {pred_path} "
                                 "Something went wrong while reading predictions")
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))

            # Do not delete the dummy prediction
            if _num_run == 1:
                continue

            numrun_dir = self.backend.get_numrun_directory(_seed, _num_run, _budget)
            try:
                os.rename(numrun_dir, numrun_dir + '.old')
                shutil.rmtree(numrun_dir + '.old')
                self.logger.info("Deleted files of non-candidate model %s", pred_path)
                self.read_losses[pred_path]["disc_space_cost_mb"] = None
                self.read_losses[pred_path]["loaded"] = 3
                self.read_losses[pred_path]["ens_loss"] = np.inf
            except Exception as e:
                self.logger.error(
                    "Failed to delete files of non-candidate model %s due"
                    " to error %s", pred_path, e
                )

    def _read_np_fn(self, path: str) -> np.ndarray:
        precision = self.precision

        if path.endswith("gz"):
            fp = gzip.open(path, 'rb')
        elif path.endswith("npy"):
            fp = open(path, 'rb')
        else:
            raise ValueError("Unknown filetype %s" % path)
        if precision == 16:
            predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float16)
        elif precision == 32:
            predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float32)
        elif precision == 64:
            predictions = np.load(fp, allow_pickle=True).astype(dtype=np.float64)
        else:
            predictions = np.load(fp, allow_pickle=True)
        fp.close()
        return predictions
