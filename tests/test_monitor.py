from contextlib import contextmanager
import os
from pathlib import Path
import pytest
import random
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

def deocrated_predict():
    @monitor(settings={"mode": "offline"})
    def predict(txt, options=None):
        return random.choice(["positive", "negative"])
    return predict

def test_run_rotation(in_tmp_path):
    """Test that the rotation works as expected."""
    orig_count = Monitor.MAX_LOGGED_COUNT
    try:
        Monitor.MAX_LOGGED_COUNT = 10
        pred = deocrated_predict()
        for i in range(15):
            pred("test", options={"step": i})
        pred.finish()
        wb_dirs = list(in_tmp_path.glob('wandb/offline-*'))
        assert len(wb_dirs) == 2, f"expected 2 runs, got {wb_dirs}"
    finally:
        Monitor.MAX_LOGGED_COUNT = orig_count
    