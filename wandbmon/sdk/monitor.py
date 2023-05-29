import atexit
import copy
import dataclasses
import datetime
import functools
import inspect
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Union, Generator
from types import FunctionType
import uuid
import wandb
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PredictionRecord:
    _timestamp: datetime.datetime
    _prediction_id: str
    _latency: float
    _args: tuple
    _kwargs: dict
    _input_preprocessor: Callable[..., Any]
    _output_postprocessor: Callable[..., Any]
    _additional_data: dict
    _output: Any = None

    def get(self):
        return self._output

    def add_data(self, data: dict) -> None:
        self._additional_data.update(data)

    @property
    def inputs(self) -> dict:
        return self._input_preprocessor(*self._args, **self._kwargs)

    @property
    def output(self) -> Any:
        logger.debug('output %s (%s)', self._output, self._output_postprocessor(self._output))
        return self._output_postprocessor(self._output)

    def as_dict(self):
        return {
            "timestamp": self._timestamp,
            "prediction_id": self._prediction_id,
            "latency": self._latency,
            **self.inputs,
            "prediction": self.output,
            **self._additional_data,
        }
    
    def __repr__(self):
        return f"PredictionRecord({self._prediction_id})"


class Monitor:
    MAX_LOGGED_COUNT = 10000
    MAX_UNSAVED_COUNT = 2
    MAX_UNSAVED_SECONDS = 5

    def __init__(self, 
        fn: Callable[..., Any],
        input_preprocessor: Optional[Callable[..., Any]] = None,
        output_postprocessor: Optional[Callable[..., Any]] = None,
        config: Union[Dict, None] = None,
        settings: Union[wandb.Settings, Dict[str, Any], None] = None,
        auto_commit: bool = True):
        functools.wraps(fn)(self)
        self.fn = fn
        self.spec = inspect.getfullargspec(fn)
        if input_preprocessor is None:
            input_preprocessor = self._default_input_processor
        if output_postprocessor is None:
            output_postprocessor = lambda x: x
        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.auto_commit = auto_commit
        self.wandb_silent: bool = True

        self.logged_count = 0
        self.last_saved_timestamp = datetime.datetime.now() - datetime.timedelta(seconds=Monitor.MAX_UNSAVED_SECONDS*2)

        self.config = config or {}
        self.settings = settings

        self.queue = queue.Queue()

        self.run: wandb.sdk.wandb_run.Run = None
        self.disabled = wandb.Api().api_key is None
        if self.disabled:
            logger.error("Monitoring disabled because WANDB_API_KEY is not set.")
            print("Couldn't find W&B API key, disabling monitoring.", file=sys.stderr)
            print("Set the WANDB_API_KEY env variable to enable monitoring.", file=sys.stderr)

        atexit.register(self._at_exit)
        self._lock = threading.Lock()
        self._join_event = threading.Event()
        self._thread = threading.Thread(target=self._thread_body)
        self._thread.daemon = True
        self._thread.start()

    def _at_exit(self):
        logger.debug('at exit called')
        self.finish()
        logger.debug('at exit done')

    def _default_input_processor(self, *args, **kwargs) -> dict:
        processed = kwargs
        if len(args) > 0:
            for i, arg in enumerate(args):
                if self.spec.args[i] == 'self':
                    continue
                processed[self.spec.args[i]] = arg
        return processed

    def _thread_body(self) -> None:
        join_requested = False
        try:
            # Initialize a run if we haven't yet
            with self._lock:
                self._ensure_run()
            while not join_requested:
                # Poll faster in the beginning
                if self.logged_count == 0:
                    wait_time = 0.1
                else:
                    wait_time = self.MAX_UNSAVED_SECONDS
                join_requested = self._join_event.wait(wait_time)
                if not self.disabled:
                    self.commit(force=join_requested)
            logger.info('monitor thread exiting cleanly')
        finally:
            wandb.finish()
            logger.info('monitor thread finally')


    def _iterate_records(self) -> Generator[PredictionRecord, None, None]:
        while True:
            try:
                record = self.queue.get_nowait()
            except queue.Empty:
                break
            else:
                yield record
                self.queue.task_done()

    # to support instance methods
    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            self.fn = self.fn.__get__(instance, owner)
            return self

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        result = self.fn(*args, **kwargs)
        duration = time.time() - start_time
        # TODO: wireup
        additional_data = {}
        pred = PredictionRecord(
            _timestamp=datetime.datetime.now(),
            _prediction_id=str(uuid.uuid4()),
            _latency=duration,
            _args=args,
            _kwargs=kwargs,
            _output=result,
            _additional_data=additional_data,
            _input_preprocessor=self.input_preprocessor,
            _output_postprocessor=self.output_postprocessor,
        )
        if self.disabled:
            return pred

        if self.auto_commit:
            logger.debug('adding record to queue %s', pred)
            self.queue.put(pred)

        return pred

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False

    @property
    def unsaved_count(self) -> int:
        return self.queue.qsize()

    def commit(self, record: Optional[PredictionRecord] = None, force: bool = False) -> None:
        """Commit all pending records to wandb.
        
        Arguments:
            record (PredictionRecord):  A single record to commit, if provided this will be added to the queue
            force (bool):  If True, commit all pending records regardless of the current state of the queue
        """
        # TODO: this lock may not be needed...
        logger.debug('monitor state %s (%s)', self.unsaved_count, self.last_saved_timestamp)

        if record is None and not force and self.unsaved_count < Monitor.MAX_UNSAVED_COUNT or self.last_saved_timestamp > datetime.datetime.now() - datetime.timedelta(seconds=Monitor.MAX_UNSAVED_SECONDS):
            logger.debug('skipping commit')
            return

        if record is not None:
            logger.debug('adding record to queue manually %s', record)
            self.queue.put(record)
            if not force:
                # TODO: maybe set the join event
                return

        with self._lock:
            logger.info('syncing %s records', self.unsaved_count)
            for r in self._iterate_records():
                self.run.log(r.as_dict())
                self.logged_count += 1
            self.last_saved_timestamp = datetime.datetime.now()
            self._maybe_rotate_run()

    def finish(self):
        self._join_event.set()
        self._thread.join()
    
    def _init_run(self) -> wandb.sdk.wandb_run.Run:
        run = wandb.init(config=self.config, settings=self.settings, job_type='monitor')
        if self.config.get("model_name", None):
            try:
                run.use_artifact(self.config["model_name"])
            except wandb.errors.CommError as e:
                # the model name might not be a valid wandb artifact, catch the error and move on
                pass
        
        if not self.wandb_silent:
            print(f'Find your run at {run.url}', file=sys.stderr)
        return run

    def _ensure_run(self) -> wandb.sdk.wandb_run.Run:
        if self.run is None:
            logger.info('initializing run')
            default_settings = {"project": "monitor", "silent": True}
            if self.settings is not None:
                self.wandb_silent = self.settings.get('silent')
                if not isinstance(self.settings, wandb.Settings):
                    default_settings.update(self.settings)
                    logger.debug('applying custom settings %s', self.settings)
                    self.settings = wandb.Settings(**default_settings)
            # TODO: silence Settings object?
            self.settings = self.settings or default_settings
            self.run = self._init_run()
        return self.run

    def _maybe_rotate_run(self) -> bool:
        if self.run is not None and self.logged_count > self.MAX_LOGGED_COUNT:
            config = dict(self.run.config)  # type: ignore[union-attr]
            settings = copy.copy(self.run._settings)  # type: ignore[union-attr]
            settings.update({"run_id": None})
            logger.info('rotating run: %s', self.run.id)
            wandb.finish()
            self._flush_count = 0
            # TODO: verify this is actually enough
            self._init_run()
            return True
        else:
            return False


def monitor(
    commit: bool = True,
    input_preprocessor: Optional[Callable[..., Any]] = None,
    output_postprocessor: Optional[Callable[..., Any]] = None,
    config: Union[Dict, str, None] = None,
    settings: Union[wandb.Settings, Dict[str, Any], None] = None):
    """Monitor a function.  This is a function decorator for performantely monitoring predictions during inference.
    We also capture system utilization, call time, and number of calls.  By default we log all keyword arguments to
    W&B.  Specify an input_preprocessor or output_postprocessor to have more control over what gets logged.  Each
    callbable should return a dictionary with scalar values or wandb Media objects.

    Arguments:
        commit (bool):  Whether to commit the run to wandb, to have more control over what data is logged set this
           to False and call .commit on the decorated function manually.
        input_preprocessor (Callable[..., Any]):  A function that takes the kwargs of the decorated function and
            returns a dictionary of key value pairs to log.
        output_postprocessor (Callable[..., Any]):  A function that takes the return value of the decorated function
            and returns a dictionary or single value to log.
        config (dict):  Sets the config parameters for the wandb.run created. If a "model_name" key is set, the wandb.run will
            attempt to use the "model_name" as a source artifact for lineage tracking.
        settings (dict, wandb.Settings):  A wandb.Settings object or dictionary to configure the run
    """

    def decorator(fn: Callable[..., Any]):
        logger.debug('decorating %s', fn)
        return Monitor(fn, input_preprocessor, output_postprocessor,
            config=config, settings=settings, auto_commit=commit)

    return decorator
