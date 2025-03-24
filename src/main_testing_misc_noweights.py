from lib import FFNN
from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

x_train = np.array([
        [0.5, -0.2, 0.1],
        [0.1, 0.4, -0.3],
        [-0.3, 0.2, 0.6],
        [0.7, -0.8, 0.9],
        [0.2, 0.5, -0.4]
    ])

target = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
])

model = FFNN.FFNN([3, 4, 3, 2], x_train, target, 0.5, [None, Activation.relu, Activation.sigmoid, Activation.sigmoid])
model.initializeWeightZeros()

for i in range(10):
    model.FFNNForwardPropagation()
    
    print(f'Epoch {i+1} Prediction:')
    print(np.array2string(model.layer_results[-1], formatter={'float_kind':lambda x: "%.6f" % x}))
    print()
    
    model.FFNNBackPropagation()