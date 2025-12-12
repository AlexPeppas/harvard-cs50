import numpy as np
import dataset.data_generator as data_gen

np.random.seed(0)

# a batch of 3, 4 dimensional vectors as an input
'''X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]'''

# let's create more realistic data of a spiral consisted of 3 different classes and 100 points each
X,y = data_gen.create_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons_per_layer):
        # scale by 0.10 to keep the values close to [-1,1] as our initial seed oscillates around 0
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons_per_layer) # now we directly modeled our weights as number of inputs * number of neurons to avoid Transposing repetitively
        self.biases = np.zeros((1, n_neurons_per_layer)) # begin with 0s as biases for now
        pass

    def forward(self, inputs, algo):
        self.outputs = np.dot(inputs, self.weights) + self.biases
        # run activation function directly on the calculated outputs of the layer
        if algo == "softmax":
            self.outputs = Activation_Softmax.forward(self.outputs)
        else:
            self.outputs = Activation_ReLu.forward(self.outputs)
    
class Activation_ReLu:
    @staticmethod
    def forward(inputs) -> np.ndarray:
        return np.maximum(0, inputs)
    
class Activation_Softmax:
    @staticmethod
    def forward(inputs) -> np.ndarray:
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))
        probabilities = exp_values / np.sum(exp_values, axis =1, keepdims= True)
        return probabilities


layer1 = Layer_Dense(2, 3)
layer2 = Layer_Dense(3, 3)
#layer3 = Layer_Dense(7,2)

layer1.forward(X, algo= "relu")
layer2.forward(layer1.outputs, algo="softmax")
#layer3.forward(layer2.outputs)

print(layer2.outputs[:5])