from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

class FFNN:
    # Static attributes

    def __init__(self, layer_neurons, input, target, learning_rate, activation_functions=None):
        self.layer_neurons = layer_neurons
        self.input = input
        self.target = target
        self.learning_rate = learning_rate
        self.activations = activation_functions

        if activation_functions != None and len(activation_functions) != len(layer_neurons):
            raise ValueError("Number of activation functions must match number of layers")

        # For weight initialization
        self.sizes = [(self.layer_neurons[i] + 1, self.layer_neurons[i + 1]) for i in range(len(self.layer_neurons) - 1)]
    
    def setLearningRate(self, learning_rate):
        self.learning_rate = learning_rate

    def setActivationUniform(self, activation_function):
        self.activations = [activation_function for i in range(len(self.layer_neurons))]

    def setWeights(self, weights):
        self.weights = weights

    def initializeWeightZeros(self):
        self.weights = [np.zeros(size) for size in self.sizes]

    def initializeWeightRandomUniform(self, lower_bound, upper_bound, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.weights = [np.random.uniform(lower_bound, upper_bound, size) for size in self.sizes]

    def initializeWeightRandomNormal(self, mean, variance, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.weights = [np.random.normal(mean, variance, size) for size in self.sizes]

    def FFNNForwardPropagation(self):
        self.layer_results = [self.input]
        self.layer_results_before_activation = [self.input]
        input = Matrix.addBiasColumn(self.input)
        i = 1
        for layer in self.weights:
            # Get layer result
            initial_result = np.matmul(input, layer)
            self.layer_results_before_activation.append(initial_result)

            # Apply activation function to result
            activation = self.activations[i]
            result = activation(initial_result)
            self.layer_results.append(result)

            # Change input to result
            biased_result = Matrix.addBiasColumn(result)
            input = biased_result
            i += 1

            # Return if at output layer
            if i > len(self.weights):
                return result

    def FFNNBackPropagation(self):
        deltas = []
        delta_weights = []
        n = len(self.weights)

        for i in range(n, 0, -1):
            # Calculate delta matrix
            output = self.layer_results[i]                                  # Output (Oj) matrix

            if i == n:
                # Output layer
                delta = (self.target - output) * Activation.getDerivativeMatrix(self.activations[i], output)
            else:
                # Hidden layer
                weight_ds = Matrix.removeBiasRow(self.weights[i])           # Downstream weight (Wkj) matrix
                delta_ds = deltas[0]                                        # Downstream delta (delta_k) matrix

                delta = Activation.getDerivativeMatrix(self.activations[i], output) * (np.matmul(delta_ds, np.transpose(weight_ds)))

            deltas = [delta] + deltas

            # Calculate new weights
            layer_input = Matrix.addBiasColumn(self.layer_results[i - 1])   # Input (Xji) matrix
            weight_change = self.learning_rate * (np.matmul(np.transpose(layer_input), delta))      # delta_w (n * delta_j * xji)

            delta_weights = [weight_change] + delta_weights

        # Update old weights after backpropagation has finished
        for i, weight_change in enumerate(delta_weights):
            self.weights[i] += weight_change

    