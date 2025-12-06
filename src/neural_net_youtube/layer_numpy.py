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
           [0.5, -0.91, .26, -0.5 ],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# [1,2,3] dot [2,0,1] => 1*2 + 2*0 + 3*1 = 5
# [[1,2,3], [1,1,1]] dot [2,0,1] => [(1*2 + 2*0 + 3*1), (1*2 + 1*0 + 1*1)] = [5, 3]
output = np.dot(inputs,weights[0]) + biases[0]

print(f'neuron output: {output}')

layer_output = np.dot(weights,inputs) + biases
print(f'layer output: {layer_output}')