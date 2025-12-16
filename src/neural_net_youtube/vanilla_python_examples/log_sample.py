import numpy as np
import math

# natural logaritm
# log(x) = y. 
# x = e^y


x = 5.2

y = np.log(x)
print(y)
print(math.e **y) # should be x

sample_softmax_output = [0.7, 0.1, 0.2]
prediction = [1, 0, 0]

loss = -math.log(sample_softmax_output[0])
print(loss)