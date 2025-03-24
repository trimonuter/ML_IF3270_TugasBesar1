import numpy as np

# Linear
@np.vectorize
def linear(x):
    return x

# ReLU
@np.vectorize
def relu(x):
    return max(0, x)

# Sigmoid
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyberbolic Tangent (tanh)
@np.vectorize
def tanh(x):
    return np.tanh(x)

# Softmax
def softmax(x):
    pass