import numpy as np

# MSE
def mse(target, prediction):
    return np.mean((target - prediction) ** 2)

# Binary Cross Entropy

# Categorical Cross Entropy