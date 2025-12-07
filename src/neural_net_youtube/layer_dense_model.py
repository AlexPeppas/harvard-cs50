import numpy as np

np.random.seed(0)

# a batch of 3, 4 dimensional vectors as an input
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons_per_layer):
        # scale by 0.10 to keep the values close to [-1,1] as our initial seed oscillates around 0
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons_per_layer) # now we directly modeled our weights as number of inputs * number of neurons to avoid Transposing repetitively
        self.biases = np.zeros((1, n_neurons_per_layer)) # begin with 0s as biases for now
        pass

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 6)
layer2 = Layer_Dense(6, 7)
layer3 = Layer_Dense(7,2)

layer1.forward(X)
layer2.forward(layer1.outputs)
layer3.forward(layer2.outputs)

print(layer3.outputs)