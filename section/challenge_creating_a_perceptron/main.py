import numpy as np
import os
os.system('wget https://codefinity-content-media.s3.eu-west-1.amazonaws.com/f9fc718f-c98b-470d-ba78-d84ef16ba45f/section_2/activations.py 2>/dev/null')
from activations import relu, sigmoid

# Fix the seed for reproducibility
np.random.seed(10)

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.inputs = np.zeros((n_inputs, 1))
        self.outputs = np.zeros((n_neurons, 1))
        # 1. Initialize the weight matrix and the bias vector with random values
        self.weights = np.random.uniform(-1, 1, (n_neurons, n_inputs))
        self.biases = np.random.uniform(-1, 1, (n_neurons,1)) 
        self.activation = activation_function
    
    def forward(self, inputs):
        self.inputs = np.array(inputs).reshape(-1, 1)
        # 2. Compute the raw output values of the neurons
        self.outputs = self.activation(np.dot(self.weights, self.inputs) + self.biases)
        # 3. Apply the activation function
        return self.outputs

class Perceptron:
    def __init__(self, layers):
        self.layers = layers

input_size = 2
hidden_size = 3
output_size = 1
# 4. Define three layers: 2 hidden layers and 1 output layer
hidden_1 = Layer(input_size, hidden_size, relu)
hidden_2 = Layer(hidden_size, hidden_size, relu)
output_layer = Layer(hidden_size,output_size, sigmoid)

layers = [hidden_1, hidden_2, output_layer]
# A perceptron with 3 layers
perceptron = Perceptron(layers)

print("Weights of the third neuron in the second hidden layer:")
print(np.round(perceptron.layers[1].weights[2], 2))

print("Weights of the neuron in the output layer:")
print(np.round(perceptron.layers[2].weights[0], 2))