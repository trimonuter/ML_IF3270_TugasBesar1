from lib import matrix as Matrix
from lib import activation as Activation
import numpy as np

class FFNN:
    def __init__(self, input, weights, target):
        self.input = input
        self.weights = weights
        self.target = target
        self.learning_rate = 0.5

    def FFNNForwardPropagation(self):
        self.layer_results = [self.input]
        input = Matrix.addBiasColumn(self.input)
        i = 1
        for layer in self.weights:
            # Get layer result
            initial_result = np.matmul(input, layer)

            # Apply activation function to result
            result = Activation.sigmoid(initial_result)
            self.layer_results.append(result)
            print(f'Layer {i}: {result}')

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
            output = self.layer_results[i]                                                           # Output (Oj) matrix

            if i == n:
                # Output layer
                delta = (self.target - output) * (output * (1 - output))
            else:
                # Hidden layer
                weight_ds = Matrix.removeBiasRow(self.weights[i])                      # Downstream weight (Wkj) matrix
                delta_ds = deltas[0]                                                                        # Downstream delta (delta_k) matrix

                delta = (output * (1 - output)) * (np.matmul(delta_ds, np.transpose(weight_ds)))

            deltas = [delta] + deltas

            # Calculate new weights
            layer_input = Matrix.addBiasColumn(self.layer_results[i - 1])         # Input (Xji) matrix
            weight_change = self.learning_rate * (np.matmul(np.transpose(layer_input), delta))      # delta_w (n * delta_j * xji)

            delta_weights = [weight_change] + delta_weights

            # Update old weights
            # self.weights[i - 1] += weight_change

        # Update old weights after backpropagation has finished
        for i, weight_change in enumerate(delta_weights):
            self.weights[i] += weight_change
