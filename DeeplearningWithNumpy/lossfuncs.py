"""
Abstracts calculation of delta in the output and gradient delta.
"""
import numpy as np
from numpy import ndarray as Tensor


class LossFunction:
    def calcOutputLoss(self, expectedOutput: Tensor, predictedOutput: Tensor) -> float:
        raise NotImplementedError

    def calcOuptutGradient(self, expectedOutput: Tensor, predictedOutput: Tensor) -> Tensor:
        raise NotImplementedError


class MeanSquaredErrorFunc(LossFunction):
    def calcOutputLoss(self, expectedOutput: Tensor, predictedOutput: Tensor) -> float:
        return np.sum((predictedOutput - expectedOutput) ** 2)

    def calcOuptutGradient(self, expectedOutput: Tensor, predictedOutput: Tensor) -> Tensor:
        return 2 * (predictedOutput - expectedOutput)
