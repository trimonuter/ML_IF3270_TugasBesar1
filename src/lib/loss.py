
import numpy as np
from lib import activation as Activation

# MSE
def mse(target, prediction):
    return np.mean((target - prediction) ** 2)

# Binary Cross Entropy
def bce(target, prediction):
    epsilon = 1e-12 # avoiding log(0) errors
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    return -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

# Categorical Cross Entropy
def cce(target, prediction):
    epsilon = 1e-12 # avoiding log(0) errors    
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    return -np.mean(np.sum(target * np.log(prediction)), axis=1)
def categorical_cross_entropy_gradient(X, y_true, W):
    input = np.dot(X, W)  # Output sebelum softmax
    y_pred = Activation.softmax(input)  # Softmax probabilities
    dL_dy = (y_pred - y_true) / X.shape[0]  
    dW = np.dot(X.T, dL_dy)
    return dW