### Softmax
# combines Exponentiation and Normalization

## Problem
# learning from clipped values can be problematic, since you don't know if it is -1.4 or -9000
# e^x > result will be positive, while we do not loose the meaning of "x" (or "-x")

import math
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]


### Exponate

#E = 2.71828182846
E = math.e

# exp_values = []
# for output in layer_outputs:
#     exp_values.append (E**output)

#more elegant with numpy:
exp_values = np.exp(layer_outputs)

print (exp_values)

### Normalize (probability distribution)
# norm_base = sum(exp_values)
# norm_values = []

# for value in exp_values:
#     norm_values. append (value / norm_base)

#more elegant with numpy:
norm_values = exp_values / np.sum(exp_values)
    
print (norm_values)
print (sum(norm_values))


# ---------------------------------------------------------

### applying Softmax with Numpy

new_layer_outputs = [[4.8, 1.21, 2.385],
    [8.9, -1.81, 0.2],
    [1.41, 1.051, 0.026]]

new_exp_values = np.exp (new_layer_outputs)

## Sum of the rows
# axis 0 > Columns, axis 1 > rows. |. Kepdims makes sure we get back an array of the same shape ([[...]])
print (np.sum(new_layer_outputs, axis=1, keepdims=True))

# actual normalized values (exponential / sum of row)
new_norm_values = new_exp_values / np.sum(new_layer_outputs, axis=1, keepdims=True)
print(new_norm_values)


### Overflow Prevention
# mitigation: subtract max value from all other values
# > makes max = 0 and all other values something negative
# > with the exponentiation this will make all values somewhere between 1 (e^0) and 1 (e^-x)