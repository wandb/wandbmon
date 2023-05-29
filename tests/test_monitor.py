from contextlib import contextmanager
import os
from pathlib import Path
import pytest
import random
from unittest.mock import MagicMock, patch
import wandb
from wandbmon import monitor, Monitor

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

@pytest.fixture
def in_tmp_path(tmp_path):
    with cd(tmp_path):
        yield tmp_path

@pytest.fixture
def wandb_mock():
    wandb_mock = MagicMock()
    wandb_mock.run = None
    wandb_mock.Settings = type({})
    wandb_mock.errors = wandb.errors
    return wandb_mock

def decorated_predict(config = {}, settings = {}):
    @monitor(config=config, settings=settings)
    def predict(txt, options=None):
        return random.choice(["positive", "negative"])
    return predict

def test_run_rotation(in_tmp_path):
    """Test that the rotation works as expected."""
    orig_count = Monitor.MAX_LOGGED_COUNT
    settings = {"mode": "offline"}
    try:
        Monitor.MAX_LOGGED_COUNT = 10
        pred = decorated_predict(settings=settings)
        for i in range(15):
            pred("test", options={"step": i})
        pred.finish()
        wb_dirs = list(in_tmp_path.glob('wandb/offline-*'))
        assert len(wb_dirs) == 2, f"expected 2 runs, got {wb_dirs}"
    finally:
        Monitor.MAX_LOGGED_COUNT = orig_count

def test_artifact_linking(wandb_mock):
    settings = {"mode": "offline"}
    config = {"model_name": "my-model"}
    with patch("wandbmon.sdk.monitor.wandb", new=wandb_mock) as wandb_mock:
        mock_run = MagicMock()
        wandb_mock.init.return_value = mock_run
        pred = decorated_predict(config, settings)
        pred("test")
        pred.finish()
        wandb_mock.init.assert_called_with(config=config, settings=settings, job_type="monitor")
        mock_run.use_artifact.assert_called_once_with("my-model")

def test_artifact_linking_non_wandb(wandb_mock):
    settings = {"mode": "offline"}
    config = {"model_name": "not-wandb-model"}
    with patch("wandbmon.sdk.monitor.wandb", new=wandb_mock) as wandb_mock:
        mock_run = MagicMock()
        mock_run.use_artifact.side_effect = wandb.errors.CommError("not an artifact")
        wandb_mock.init.return_value = mock_run
        pred = decorated_predict(config, settings)
        pred("test")
        pred.finish()
        wandb_mock.init.assert_called_with(config=config, settings=settings, job_type="monitor")
        mock_run.use_artifact.assert_called_once()
        mock_run.log.assert_called_once()
