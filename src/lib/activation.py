import numpy as np

# Linear

# ReLU

# Sigmoid
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyberbolic Tangent (tanh)

# Softmax