import logging
import random
import sys
from typing import List
import wandb
from wandbmon import monitor, configure_logging

class Model:
    def __init__(self, name: str) -> None:
        self.name = name
        self.predict.config.update({"model_name": name})

    @monitor(
        input_preprocessor=lambda x: {"inputs": wandb.Histogram(x)},
        settings={"project": "wandbmon"}
    )
    def predict(self, inputs: List[float]) -> float:
        return sum(inputs) / len(inputs)

if __name__ == "__main__":
    configure_logging(logging.DEBUG)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    model = Model("test-hf:v0")
    for i in range(100):
        model.predict([random.random() for _ in range(100)])
