from lib import matrix as Matrix
from lib import activation as Activation
from lib import loss as Loss
from lib import color as Color
import numpy as np

class FFNN:
    # Static attributes

    def __init__(self, layer_neurons, X_train, y_train, learning_rate, activation_functions=None, X_val=None, y_val=None):
        self.layer_neurons = layer_neurons
        self.input = X_train
        self.target = y_train
        self.learning_rate = learning_rate
        self.activations = activation_functions

        if activation_functions != None and len(activation_functions) != len(layer_neurons):
            raise ValueError("Number of activation functions must match number of layers")

        # For weight initialization
        self.sizes = [(self.layer_neurons[i] + 1, self.layer_neurons[i + 1]) for i in range(len(self.layer_neurons) - 1)]

        # For validation
        self.X_val = X_val
        self.y_val = y_val
    
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

    def FFNNForwardPropagation(self, current_input):
        self.layer_results = [current_input]
        self.layer_results_before_activation = [current_input]
        input = Matrix.addBiasColumn(current_input)
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

    def FFNNBackPropagation(self, current_target):
        deltas = []
        delta_weights = []
        n = len(self.weights)

        for i in range(n, 0, -1):
            # Calculate delta matrix
            output = self.layer_results[i]                                  # Output (Oj) matrix

            if i == n:
                # Output layer
                delta = (current_target - output) * Activation.getDerivativeMatrix(self.activations[i], output)
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

        # Set current epoch's gradient array
        self.deltas = deltas

        # Update old weights after backpropagation has finished
        for i, weight_change in enumerate(delta_weights):
            self.weights[i] += weight_change

    def train(self, batch_size, learning_rate, epochs, verbose=True, printResults=False):
        self.setLearningRate(learning_rate)
        training_loss_list = []
        validation_loss_list = []

        for epoch in range(epochs):
            training_loss = 0
            validation_loss = 0
            
            # Iterate through batches
            for i in range(0, len(self.input), batch_size):
                batch_end = (i + batch_size) if (i + batch_size) < len(self.input) else len(self.input)
                X_batch = self.input[i:batch_end]
                y_batch = self.target[i:batch_end]

                self.FFNNForwardPropagation(X_batch)
                self.FFNNBackPropagation(y_batch)

                training_loss += Loss.mse(y_batch, self.layer_results[-1])

            # Calculate epoch loss
            training_loss /= len(self.input)
            training_loss_list.append(training_loss)

            if self.X_val != None and self.y_val != None:
                self.FFNNForwardPropagation(self.X_val)
                validation_loss = Loss.mse(self.y_val, self.layer_results[-1])
                validation_loss_list.append(validation_loss)

            # Print epoch results
            if verbose:
                progress_bar = Color.progress_bar(epoch + 1, epochs)
                print(Color.YELLOW + f" [Epoch {epoch + 1}]:" + Color.GREEN + f"\tTraining Loss: {training_loss}" + Color.BLUE + f"\tValidation Loss: {validation_loss}" + Color.YELLOW + f'\tProgress: [{progress_bar}]' + Color.RESET)
                if printResults:
                    print(f"{Color.CYAN}     Prediction:\t{self.layer_results[-1]}")
                    print(f"{Color.MAGENTA}     Target:\t\t{self.target}{Color.RESET}")

        return training_loss_list, validation_loss_list