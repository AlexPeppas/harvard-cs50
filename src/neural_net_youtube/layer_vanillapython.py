inputs = [1,2,3,2.5]

# the weight amplifies (positively,negatively) the magnitude of the input
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, .26, -0.5 ],
           [-0.26, -0.27, 0.17, 0.87]]

# the bias offsets it
biases = [2, 3, 0.5]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for weight, input in zip(neuron_weights, inputs):
        neuron_output+= weight*input
    neuron_output+=neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)