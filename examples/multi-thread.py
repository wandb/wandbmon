import random
import threading
from typing import List
import wandb
from wandbmon import monitor

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
    def pred_20():
        model = Model("test")
        thread_id = threading.current_thread().ident
        # This is stupid but I'm curious what happens
        model.predict.config.update({f"{thread_id}": thread_id})
        for i in range(20):
            model.predict([random.random() for _ in range(100)])
        # TODO: maybe call commit?
    threads = [threading.Thread(target=pred_20) for _ in range(5)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()