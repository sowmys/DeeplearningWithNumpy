"""
Abstracts various layers used in deep learning typically such as Linear Layer, Activation Layer etc.
"""
from typing import Dict

import numpy as np
from numpy import ndarray as Tensor

from DeeplearningWithNumpy.optimizers import Optimizer


class Layer:
    """
    Transforms input vector to an output vector during forward propagation using its parameters that is learned during
    training
    Transforms gradient vector of the output (difference between expected and actual output of the layer) to gradients
    that can be applied each of the learned parameters during backward propagation
    """
    __name: str
    __learnedParameters: Dict[str, Tensor]
    __gradients: Dict[str, Tensor]

    def __init__(self, name: str) -> None:
        self.__learnedParameters = dict()
        self.__gradients = dict()
        self.__name = name

    def propagate_forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def propagate_backward(self, gradient: Tensor) -> Tensor:
        raise NotImplementedError

    def update_parameters(self, optimizer: Optimizer) -> None:
        for learned_param_name, learned_param_value in self.__learnedParameters.items():
            learned_param_gradient = self.__gradients[learned_param_name]
            optimizer.update_parameters(learned_param_value, learned_param_gradient)

    def get_parameter(self, name: str) -> Tensor:
        return self.__learnedParameters[name]

    def set_parameter(self, name: str, value: Tensor) -> None:
        self.__learnedParameters[name] = value

    def get_gradient(self, name: str) -> Tensor:
        return self.__gradients[name]

    def set_gradient(self, name: str, value: Tensor) -> None:
        self.__gradients[name] = value

    def print(self) -> None:
        print("Layer = ", self.__name)
        print("  Params")
        for name, value in self.__learnedParameters.items():
            print("    " + name, " = ", list(value))
        print("  Grads")
        for name, value in self.__gradients.items():
            print("    " + name, " = ", list(value))


class LinearLayer(Layer):
    """
    The linear layer has two main parameters: weights (W), biases (b)
    During forward propagation: Output = Input @ W + b
    During backward propagation:

    """
    __inputs: Tensor
    __W: str = "W"
    __b: str = "b"

    def __init__(self, name: str, input_size: int, output_size: int) -> None:
        super().__init__(name)
        self.set_parameter(self.__W, np.random.randn(input_size, output_size))
        self.set_parameter(self.__b, np.random.randn(output_size))

    def propagate_forward(self, inputs: Tensor) -> Tensor:
        self.__inputs = inputs
        return inputs @ self.get_parameter(self.__W) + self.get_parameter(self.__b)

    def propagate_backward(self, gradient: Tensor) -> Tensor:
        self.set_gradient(self.__b, np.sum(gradient, axis=0))
        self.set_gradient(self.__W, self.__inputs.T @ gradient)
        return gradient @ self.get_parameter(self.__W).T


class Activation(Layer):
    """
    An activation layer just applies a function
    element-wise to its inputs
    """
    inputs: Tensor

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def propagate_forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.activator_fn(inputs)

    def propagate_backward(self, gradients: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.activator_prime_fn(self.inputs) * gradients

    def activator_fn(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def activator_prime_fn(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class TanhActivator(Activation):
    def __init__(self, name: str):
        super().__init__(name)

    def activator_fn(self, x: Tensor) -> Tensor:
        return np.tanh(x)

    def activator_prime_fn(self, x: Tensor) -> Tensor:
        y = np.tanh(x)
        return 1 - y ** 2
