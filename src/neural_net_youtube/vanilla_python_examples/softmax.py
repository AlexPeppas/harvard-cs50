import math
import nnfs

nnfs.init()

E = math.e
layer_outputs = [4.8, 1.21, 2.385]

expo_values = []

for output in layer_outputs:
    expo_values.append(E**output)

print(expo_values)

normalized_values = []
norm_sum = sum(expo_values)

for value in expo_values:
    normalized_values.append(value/norm_sum)

# this is the normalized probability distribution using exponentiation to tackle negative values and normalization
# Normalization => y = u / Σ un, where y is the normalized output of the neuron by dividing its original value by the sum of output values of the entire layer 
print(f'normalized values: {normalized_values}')

# This process of exponentiation and normalization is called Softmax