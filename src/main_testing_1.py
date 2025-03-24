from lib import FFNN
from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

network = [
    np.array(
        [[0.35, 0.35],
        [0.15, 0.25],
        [0.20, 0.30]]),
    np.array(
        [[0.60, 0.60],
        [0.40, 0.50],
        [0.45, 0.55]]),
]
X = np.array([[0.05, 0.10]])
target = np.array([[0.01, 0.99]])

model = FFNN.FFNN([2, 2, 2], X, target, 0.5, Activation.sigmoid)
model.setWeights(network)

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