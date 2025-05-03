from dataclasses import dataclass
from typing import Optional
import random

import numpy as np


@dataclass
class Observation:
    features: np.ndarray
    values: np.ndarray


class Perceptron:
    def __init__(
        self,
        initial_weights: list,
        initial_bias: Optional[list] = None,
        eta: float = 0.1,
    ) -> None:
        self.weights = np.array(initial_weights)
        if initial_bias is not None:
            self.bias: Optional[np.ndarray] = np.array(initial_bias)
        else:
            self.bias = None
        self.eta = np.array(eta)

    def activity_value(self, input_vector: np.ndarray) -> np.ndarray:
        # row vector * column vector + bias
        return input_vector @ self.weights + (self.bias if self.bias is not None else 0)

    def activation_value(self, input_vector: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.activity_value(input_vector)))

    def compute_error(self, desired_output: np.ndarray) -> np.ndarray:
        self.error = desired_output - self.last_output
        return self.error

    def prev_delta(self) -> np.ndarray:
        delta = self.weights * self.delta.reshape(-1, self.weights.shape[1])
        return delta

    def compute_delta(self, error: np.ndarray) -> np.ndarray:
        self.delta: np.ndarray = (
            error.reshape(-1, self.weights.shape[1])
            * (1 - self.last_output)
            * self.last_output
        )
        return self.delta

    def feed_forward(self, input_vector: np.ndarray) -> np.ndarray:
        self.last_input = input_vector
        self.last_output = self.activation_value(input_vector)
        return self.last_output

    def back_prop(self, error: np.ndarray) -> np.ndarray:
        """returns the deltas to be used by the previous layer and updates model
        weights"""
        self.compute_delta(error)
        prev_delta = self.prev_delta()
        self.update_weights()

        return prev_delta

    def update_weights(self) -> None:
        self.weights = self.weights + self.eta * (self.delta * self.last_input.T)

        if self.bias is not None:
            self.bias = self.bias + self.eta * self.delta

    def __call__(self, input_vector: np.ndarray) -> float:
        return self.activation_value(input_vector)[0][0]


class Multilayer:
    def __init__(
        self, initial_weights: list, initial_bias: list, eta: float = 0.1
    ) -> None:
        self.layers = [
            Perceptron(weights, biases, eta)
            for weights, biases in zip(initial_weights, initial_bias)
        ]

    def activation_value(self, input_vector: np.ndarray) -> np.ndarray:
        data_in = input_vector
        for layer in self.layers:
            out = layer.activation_value(data_in)
            data_in = out.reshape(1, -1)  # convert row vector to column
        return out

    def feed_forward(
        self, input_vector: np.ndarray, desired_ouput: np.ndarray
    ) -> np.ndarray:
        data_in = input_vector
        for layer in self.layers:
            out = layer.feed_forward(data_in)
            data_in = out.reshape(1, -1)

        self.error = desired_ouput - out
        return out

    def backwards(self) -> None:
        err = self.error
        # reversing layers
        for layer in self.layers[::-1]:
            err = layer.back_prop(err)

    def __call__(self, input_vector: np.ndarray) -> float:
        return self.activation_value(input_vector)[0][0]


def big_e(model: Multilayer | Perceptron, observation: Observation) -> float:
    return (
        model.activation_value(observation.features)[0][0] - observation.values[0][0]
    ) ** 2 / 2


def init_multilayer_model(
    input_size: int,
    hidden_nodes: int,
    default_weights: Optional[list] = None,
    default_biases: Optional[list] = None,
    include_biases: bool = True,
    eta: float = 0.1,
) -> Multilayer:
    if default_weights is not None:
        weights = default_weights
    else:
        weights = [
            # hidden layer weights
            # height equal to number of inputs
            # width equal to number of hidden nodes
            [[random.uniform(-1, 1) for _ in range(hidden_nodes)] for _ in range(input_size)],
            # output layer weights
            # height equal to number of hidden nodes
            # width equal to number of output nodes (1)
            [[random.uniform(-1, 1)] for _ in range(hidden_nodes)],
        ]

    if default_biases is not None:
        biases = default_biases
    elif include_biases:
        biases = [
            # hidden layer biases
            # height is 1
            # width equal to number of hidden nodes
            [[random.uniform(-1, 1) for _ in range(hidden_nodes)]],
            # output layer bias
            # height is 1
            # width equal to number of output nodes (1)
            [[random.uniform(-1, 1)]],
        ]
    else:
        biases = [None, None]

    return Multilayer(weights, biases, eta)


def init_perceptron_model(
    input_size: int,
    default_weights: Optional[list] = None,
    default_biases: Optional[list] = None,
    include_bias: bool = True,
    eta: float = 0.1,
) -> Perceptron:
    if default_weights is not None:
        weights = default_weights
    else:
        weights = [[random.uniform(-1, 1)] for _ in range(input_size)]

    if default_biases is not None:
        bias = default_biases
    elif include_bias:
        bias = [[random.uniform(-1, 1)]]
    else:
        bias = None

    return Perceptron(weights, bias, eta)


def build_observations(inputs: list[list]) -> list[Observation]:
    return [
        Observation(np.array([input[:-1]]), np.array([[input[-1]]])) for input in inputs
    ]


def online_multilayer(
    model: Multilayer, training_data: list[Observation], cycles: int
) -> None:
    for _ in range(cycles):
        for ob in training_data:
            model.feed_forward(ob.features, ob.values)
            model.backwards()


def online_perceptron(
    model: Perceptron, training_data: list[Observation], cycles: int
) -> None:
    for _ in range(cycles):
        for ob in training_data:
            model.feed_forward(ob.features)
            model.back_prop(model.compute_error(ob.values))

def predict_with_threshold(model, input_vector: np.ndarray, threshold: float = .5) -> int:
    output = model(input_vector)
    return int(output >= threshold)
