"""
Abstracts the optimizers that is used to update the parameters of the layers that are being learnt
"""

from numpy import ndarray as Tensor


class Optimizer:
    def update_parameters(self, parameters: Tensor, gradients: Tensor):
        raise NotImplementedError


class StatisticalGradientDescent (Optimizer):
    __learning_rate: float

    def __init__(self, learning_rate: float = 0.01):
        self.__learning_rate = learning_rate

    def update_parameters(self, parameters: Tensor, gradients: Tensor):
        parameters -= self.__learning_rate * gradients

