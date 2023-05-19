import logging
import random
import numpy as np
import wandb
from wandbmon import monitor, configure_logging

def preprocess(image):
    return {"image": wandb.Image(image)}

def postprocess(image):
    return wandb.Image(image)

@monitor(
    input_preprocessor=preprocess,
    output_postprocessor=postprocess,
    settings={"project": "wandbmon-img"},
    config={"model_name": "test"},
    commit=False)
def predict(image):
    return image / 2


if __name__ == "__main__":
    for i in range(10):
        img = np.random.randint(255, size=(28, 28, 3))
        res = predict(img)
        res.add_data({"step": i})
        predict.commit(res)
        print(res)
    predict.finish()
