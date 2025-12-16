import numpy as np

# probability distribution of layer N
softmax_outputs_batch = np.array([[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]])

#assuming that classes are 
# 0 - cat
# 1 - dog
# 2 - human
class_map = {
    0 : "cat",
    1 : "dog",
    2 : "human"
}

# this is a scalar example
class_targets = [0,1,1]
# its equivalent to one hot encoding --> [[1,0], [0,1], [0,1]]

# now that softmax batch is a numpy array we can directly use its idnex accessor 
# with range of length of the batch or instead simply [0,1,2] which gives the index per row in the batch
# and the class targets which dictates the index of the desired value IN the corresponding current row.
# this acts like zip()
distribution = softmax_outputs_batch[range(len(softmax_outputs_batch)), class_targets]
print (distribution) # should output [0.7, 0.5, 0.9]
# print (softmax_outputs_batch[[0,1,2], class_targets]) # should output [0.7, 0.5, 0.9]
# print (softmax_outputs_batch[[0,1,1], class_targets]) # should output [0.7, 0.5, 0.5]
# print (softmax_outputs_batch[[0,1], class_targets]) # should throw shape mismatch
# print (softmax_outputs_batch[[0,1,2, 3], class_targets]) # should throw shape mismatch

loss = -np.log(distribution)
print(f'loss distribution {loss}')

avg_loss_layer = np.mean(loss)
print(f'mean loss {avg_loss_layer}')