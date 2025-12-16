# probability distribution of layer N
softmax_outputs_batch = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

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

for class_index, distribution in zip(class_targets, softmax_outputs_batch):
    print(f'predicted value for corresponding class [{class_map[class_index]}] is {distribution[class_index]}')