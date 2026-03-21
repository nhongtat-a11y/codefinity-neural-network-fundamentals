import numpy as np
import os
os.system('wget https://codefinity-content-media.s3.eu-west-1.amazonaws.com/f9fc718f-c98b-470d-ba78-d84ef16ba45f/section_2/utils.py 2>/dev/null')
from utils import relu, sigmoid, X_train, y_train

np.random.seed(10)

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.inputs = np.zeros((n_inputs, 1))
        self.outputs = np.zeros((n_neurons, 1))
        self.weights = np.random.uniform(-1, 1, (n_neurons, n_inputs))
        self.biases = np.random.uniform(-1, 1, (n_neurons, 1))
        self.activation = activation_function

    def forward(self, inputs):
        self.inputs = np.array(inputs).reshape(-1, 1)
        self.outputs = np.dot(self.weights, self.inputs) + self.biases
        return self.activation(self.outputs)
    
    def backward(self, da, learning_rate):
        # 1. Compute the gradients
        dz = da * self.activation.derivative(self.outputs)
        d_weights = dz @ self.inputs.T
        d_biases = dz
        da_prev = self.weights.T @ dz
    
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
    
        return da_prev

class Perceptron:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def fit(self, training_data, labels, epochs, learning_rate):
        for epoch in range(epochs):
            loss = 0
            indices = np.random.permutation(training_data.shape[0])
            training_data = training_data[indices]
            labels = labels[indices]

            for i in range(training_data.shape[0]):
                inputs = training_data[i, :].reshape(-1, 1)
                target = labels[i, :].reshape(-1, 1)
                # 2. Compute the `output` of the model
                output = self.forward(inputs)
                loss += -(target * np.log(output) + (1 - target) * np.log(1 - output))
                # 3. Compute da^n
                da = (output - target) / (output * (1 - output))
                for layer in self.layers[::-1]:
                    # 4. Compute da^l-1 by calling the appropriate method
                    da = layer.backward(da, learning_rate)
            average_loss = loss[0, 0] / training_data.shape[0]
            print(f'Loss at epoch {epoch + 1}: {average_loss:.3f}')


input_size = 2
hidden_size = 6
output_size = 1

h1 = Layer(input_size, hidden_size, relu)
h2 = Layer(hidden_size, hidden_size, relu)
output_layer = Layer(hidden_size, output_size, sigmoid)

layers = [h1, h2, output_layer]
model = Perceptron(layers)
model.fit(X_train, y_train, epochs=10, learning_rate=0.01)