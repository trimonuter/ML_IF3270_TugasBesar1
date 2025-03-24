from lib import FFNN
from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

X = np.array([[1, 0]])
network = [
    np.array(
        [[0.5, 0],
        [1, 0.5],
        [1, 0.5]]),
    np.array(
        [[0.5],
        [1],
        [1]]
    ),
]
target = np.array([[2]])

model = FFNN.FFNN([2, 2, 1], X, target, 0.5)
model.setActivationUniform(Activation.sigmoid)
model.setWeights(network)
model.FFNNForwardPropagation()

print(f'Layer inputs before activation: {model.layer_results_before_activation}')
print(f'Layer inputs: {model.layer_results}')
# print(f'Layer_outputs: {[Matrix.addBiasColumn(x) for x in model.layer_results]}')
print('Weights (before backpropagation)')
for i , W in enumerate(model.weights):
    print(f'Layer {i + 1}')
    print(np.array2string(W) + '\n')

model.FFNNBackPropagation()
print('Weights (after backpropagation)')
for i , W in enumerate(model.weights):
    print(f'Layer {i + 1}')
    print(np.array2string(W) + '\n')