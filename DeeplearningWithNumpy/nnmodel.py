"""
Abstracts a neural network as a collection of layers
"""
from typing import Sequence, Iterator, Tuple
from DeeplearningWithNumpy.layers import Layer
from DeeplearningWithNumpy.optimizers import Optimizer
from DeeplearningWithNumpy.lossfuncs import LossFunction
from DeeplearningWithNumpy.dataiterators import DataIterator
from numpy import ndarray as Tensor


class NNModel:
    __layers: Sequence[Layer]
    __lossFunc: LossFunction
    __optimizer: Optimizer

    def __init__(self, layers: Sequence[Layer], loss_func: LossFunction, optimizer: Optimizer):
        self.__lossFunc = loss_func
        self.__optimizer = optimizer
        self.__layers = layers

    def __propagate_forward(self, inputs: Tensor) -> Tensor:
        for layer in self.__layers:
            inputs = layer.propagate_forward(inputs)
        return inputs

    def __propagate_back(self, gradients: Tensor) -> Tensor:
        for layer in reversed(self.__layers):
            gradients = layer.propagate_backward(gradients)
        return gradients

    def __update_parameters(self) -> None:
        for layer in self.__layers:
            layer.update_parameters(self.__optimizer)

    def train(self, num_epochs: int, training_data_iterator: DataIterator) -> None:
        for epoch in range(num_epochs):
            epoch_loss: float = 0.0
            for batch in training_data_iterator():
                predicted: Tensor = self.__propagate_forward(batch.Inputs)
                epoch_loss += self.__lossFunc.calcOutputLoss(batch.ExpectedOutput, predicted)
                gradient: Tensor = self.__lossFunc.calcOuptutGradient(batch.ExpectedOutput, predicted)
                self.__propagate_back(gradient)
                self.__update_parameters()
                if (epoch % (num_epochs/10)) == 0:
                    print(epoch, "loss = ", epoch_loss, "predicted = ", list(predicted))
                    # self.print()

    def predict(self, inputs: Tensor) -> Tensor:
        return self.__propagate_forward(inputs)

    def print(self) -> None:
        for layer in self.__layers:
            layer.print()
