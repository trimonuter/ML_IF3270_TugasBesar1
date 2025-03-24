from lib import FFNN
from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

X = np.array([[0.05, 0.10]])
target = np.array([[0.01, 0.99]])

model = FFNN.FFNN([2, 2, 2], X, target, 0.5)
model.setActivationUniform(Activation.sigmoid)
model.initializeWeightZeros()

for i in range(10000):
    model.FFNNForwardPropagation()
    
    print(f'Epoch {i+1} Prediction:')
    print(np.array2string(model.layer_results[-1], formatter={'float_kind':lambda x: "%.6f" % x}))
    print()
    
    model.FFNNBackPropagation()