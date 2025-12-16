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

    def forward(self, inputs, algo = None):
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
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        return probabilities

class Loss:
    def calculate(self, layer_outputs, y) -> int:
        batch_loss = self.forward(layer_outputs, y)
        mean_batch_loss = np.mean(batch_loss)
        return mean_batch_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_predicted , y_true):
        samples = len(y_predicted)
        # clip will put upper and lower boundaries in the layer output predicted values
        # that's in order to guard against a confident prediction of 0 because np.log(0) will return infinite.
        # Thus in the Loss.Calculate it will cause np.mean to always return infinite. 
        # 1e-7 is very close to 0 and 1-1e-7 is very close to 1.
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: # the expected true values are in scalar shape
            correct_confidences = y_predicted_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # the expected true values are in one-hot-encoded vectors
            correct_confidences = np.sum(y_predicted_clipped * y_true, axis = 1)
        else:
            raise Exception(f'predicted values shape is not supported. Len:{len(y_true)}')
        
        batch_loss = -np.log(correct_confidences)
        return batch_loss
    

layer1 = Layer_Dense(2, 3)
layer2 = Layer_Dense(3, 3)
#layer3 = Layer_Dense(7,2)

layer1.forward(X)
layer2.forward(layer1.outputs, algo="softmax")
#layer3.forward(layer2.outputs)

print(layer2.outputs[:5])

cce_loss = Loss_CategoricalCrossEntropy()

# calcualte the loss of last layer's output against the y true values 
neural_net_loss = cce_loss.calculate(layer2.outputs, y)

print (f'Overall mean loss: {neural_net_loss}')