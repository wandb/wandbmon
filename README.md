# WandMon

WandMon is a lightweight Python library designed to help you seamlessly monitor the inputs and outputs of your machine learning model in a production environment. Developed by Weights and Biases, it offers the capability to stream the monitoring data directly to your Weights and Biases dashboard.

Wrap any prediction function with a `@monitor` decorator and WandMon provides real-time tracking of your function calls with minimal interference to your existing pipeline.

## Key Features

- **Real-time monitoring**: WandMon captures real-time system utilization, call time, and number of calls.
- **Flexibility**: Easily specify an input preprocessor or output postprocessor to have more control over what gets logged.
- **Direct logging to Weights and Biases**: By default, WandMon logs all keyword arguments.
- **Dashboard integration**: WandMon is fully compatible with Weights and Biases' workspaces and integrates with [Weave](https://github.com/wandb/weave) enabling easy analysis and customization.

## Installation

You can install WandMon via pip:

```
pip install -e .
```

## Getting Started

To get started with WandMon, simply import the library and use the `@monitor` decorator for your function. WandMon will automatically log the inputs and outputs of your function to your Weights and Biases account.

See [examples](./examples) for some getting started examples.

```python
from wandmon import monitor

@monitor
def my_function(a, b, c):
    ...
```

Once you've set up WandMon, calling `my_function` will automatically stream its inputs and outputs to Weights and Biases:

```python
my_function(1, 2, 3)
```

For more control over what data is logged, you can provide an input_preprocessor or output_postprocessor:

```python
def my_input_preprocessor(kwargs):
    return {key: value for key, value in kwargs.items() if key in ["a", "b"]}

def my_output_postprocessor(result):
    return {"result": result}

@monitor(input_preprocessor=my_input_preprocessor, output_postprocessor=my_output_postprocessor)
def my_function(a, b, c):
    ...
```

## Weights and Biases Integration

To view the logged data, visit your Weights and Biases dashboard. The dashboard displays real-time system utilization, call time, and inputs and outputs.

WandMon makes it easier to monitor your production ML models and visualize data to get insights.

## Feedback and Contribution

We would love to hear your feedback and are open to contributions. Please feel free to open an issue or send a pull request.

## Development

```shell
pip install -r requirements.test.txt
```

## License

WandMon is under the [Apache 2.0](LICENSE).

---

Whether you're tracking your models' performance over time, comparing different models, or trying to diagnose the cause of a recent crash, WandMon and Weights and Biases have you covered. Let's get your models monitored, understood, and improved!
