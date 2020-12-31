from DeeplearningWithNumpy.nnmodel import NNModel
from DeeplearningWithNumpy.layers import LinearLayer, TanhActivator
from DeeplearningWithNumpy.dataiterators import BatchIterator
from DeeplearningWithNumpy.lossfuncs import MeanSquaredErrorFunc
from DeeplearningWithNumpy.optimizers import StatisticalGradientDescent
import numpy as np

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

expectedOutput = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

nnModel = NNModel([
    LinearLayer("LLayer 1", input_size=2, output_size=2),
    TanhActivator("TanHLayer"),
    LinearLayer("LLayer 2", input_size=2, output_size=2)
],
    loss_func=MeanSquaredErrorFunc(),
    optimizer=StatisticalGradientDescent()
)

training_data_iterator = BatchIterator(inputs, expectedOutput, batch_size=32, shuffle=False)
nnModel.train(num_epochs=5000, training_data_iterator=training_data_iterator)
print("result = ", list(nnModel.predict(inputs)))

