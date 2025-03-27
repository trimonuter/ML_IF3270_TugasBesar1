
import numpy as np

# MSE
def mse(target, prediction):
    return np.mean((target - prediction) ** 2)

# Binary Cross Entropy
def bce(target, prediction):
    epsilon = 1e-12 # avoiding log(0) errors
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    return -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

# Categorical Cross Entropy
def cce(target, prediction, n):
    epsilon = 1e-12 # avoiding log(0) errors    
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    return -np.mean(np.sum(target * np.log(prediction)), axis=1)