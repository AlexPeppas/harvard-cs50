import numpy as np


layer_outputs = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

expo_values = np.exp(layer_outputs)
layer_sum_per_batch = np.sum(expo_values, axis = 1, keepdims=True)
expo_values = expo_values / layer_sum_per_batch

print(f'softmax: {expo_values}')

expo_values = np.exp(layer_outputs - np.max(layer_outputs, axis = 1, keepdims=True))
layer_sum_per_batch = np.sum(expo_values, axis = 1, keepdims=True)
expo_values = expo_values / layer_sum_per_batch

# the reason we do max batch negation of the entire batch is to ensure we do not run in overflows
# for instance in a vector [1000, 999, 1] ==> e^1000 would run in an overflow
# by subtracting the max, we'll have [0, -1, -999] which then exponentially e^i will oscillate between [0,1]
# that guards us against overflows and the outputs are identical.
print(f'softmax with max negation: {expo_values}')
