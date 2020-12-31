"""
Data iterators provides component to slice input (and the corresponding expected-output) to be used for training
"""
from numpy import ndarray as Tensor
import numpy as np
from typing import Iterator, NamedTuple

TrainingDataInputOutputPair = NamedTuple("Batch", [("Inputs", Tensor), ("ExpectedOutput", Tensor)])


class DataIterator:
    expected_outputs: Tensor
    inputs: Tensor

    def __init__(self, inputs: Tensor, expected_outputs: Tensor):
        self.inputs = inputs
        self.expected_outputs = expected_outputs

    def __call__(self) -> Iterator[TrainingDataInputOutputPair]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    __shuffle: bool
    __batch_size: int

    def __init__(self, inputs: Tensor, expected_outputs: Tensor, batch_size: int, shuffle: bool):
        super().__init__(inputs, expected_outputs)
        self.__batch_size = batch_size
        self.__shuffle = shuffle

    def __call__(self) -> Iterator[TrainingDataInputOutputPair]:
        starts: Tensor = np.arange(0, len(self.inputs), self.__batch_size)
        if self.__shuffle:
            np.random.shuffle(starts)

        for start_index in starts:
            end_index = start_index + self.__batch_size
            inputs_slice: Tensor = self.inputs[start_index:end_index]
            expected_output_slice: Tensor = self.expected_outputs[start_index:end_index]
            yield TrainingDataInputOutputPair(inputs_slice, expected_output_slice)
