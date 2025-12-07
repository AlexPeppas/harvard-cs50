# [1,2,3,4] 
# Shape is 4 because it describes how many elements the list contains
# 
# [[1,2,3,4], [4,5,6,7]] lol
# Shape is (2,4) because it describes how many elements each list of lists contain
#
# [[[1,2,3,4], [4,5,6,7]], [[1,2,3,4], [4,5,6,7]]]
# Shape is (2,2,4) describing the elements for list of lists of lists 

import numpy as np

inputs = [1,2,3,2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5 ],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# [1,2,3] dot [2,0,1] => 1*2 + 2*0 + 3*1 = 5
# [[1,2,3], [1,1,1]] dot [2,0,1] => [(1*2 + 2*0 + 3*1), (1*2 + 1*0 + 1*1)] = [5, 3]
output = np.dot(inputs,weights[0]) + biases[0]

print(f'neuron output: {output}')

layer_output = np.dot(weights,inputs) + biases
print(f'layer output: {layer_output}')


### Batching

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# matrix multiplication e.g. inputs dot weights
# => [inputs[0] row vector dot weights[0] col vector, inputs[0] row vector  dot weights[1] col vector,  inputs[0] row vector dot weights[2] col vector] 
#    [ ... , ..., ...]
#    repeat for all other inputs rows 

# Transpose is when you take a matrix X and convert all of its rows to columns into a X^T
# Explanation,
# Weights is a Matrix [3,4]
# Inputs is now a Matrix [3,4]
# in matrix dot we want Shape of index 0 at first element to match index 1 at second element
# So inputs[0] = 3 dimensions while weights[1] = 4 dimensions thus we'll have a Shape error
# Once  we transpose weights we get a Matrix[4,3]
# Finally inputs[0] = 3 and weights[1] = 3 dimensions which makes the matrix dot possible again. 
transposed_weights = np.array(weights).T
batch_output = np.dot(inputs, transposed_weights) + biases
print(f'batch output: {batch_output}')