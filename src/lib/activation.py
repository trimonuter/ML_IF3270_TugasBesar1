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

# Function to apply derivative of activation function to a matrix
def getDerivativeMatrix(activation, matrix):
    if activation == linear:
        return np.ones(matrix.shape)
    elif activation == relu:
        return np.vectorize(lambda x: 1 if x > 0 else 0)(matrix)
    elif activation == sigmoid:
        return matrix * (1 - matrix)
    elif activation == tanh:
        return 1 - matrix ** 2
    elif activation == softmax:
        pass
    else:
        raise ValueError("Activation function not recognized")