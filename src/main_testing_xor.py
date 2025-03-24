from lib import FFNN
from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

# network = [
#     np.array(
#         [[-10.0, 30.0],
#         [20.0, -20.0],
#         [20.0, -20.0]]),
#     np.array(
#         [[-30.0],
#         [20.0],
#         [20.0]]
#     ),
# ]
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])

model = FFNN.FFNN([2, 2, 1], X, target, 0.5, Activation.sigmoid)
model.initializeWeightZeros()
# model.setWeights(network)

for i in range(10):
    model.FFNNForwardPropagation()
    
    print(f'Epoch {i+1} Prediction:')
    print(np.array2string(model.layer_results[-1], formatter={'float_kind':lambda x: "%.6f" % x}))

    print(f'Epoch {i+1} Weights:')
    for j, W in enumerate(model.weights):
        pass
        print(f'Layer {j + 1}')
        print(np.array2string(W, formatter={'float_kind':lambda x: "%.6f" % x}) + '\n')
    
    model.FFNNBackPropagation()