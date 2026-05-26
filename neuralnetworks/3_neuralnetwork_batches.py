### Batch processing
# useful to parallelize processing (thousands of cores on gpu compared to maybe 16-48 in cpu)

## Generalization 
# average out data (and errors) by analyzing (e.g.) 64 points in parallel
# problem: "over-fitting" > when batch size is too large we train exactly to the input data that may not be able to generalize outside of the training data

## Matrix product
# take first row from Matrix A and multiply with values form column of Matrix B
# r[0]*c[0] + r[1]*c[1] ... = Skalar (first value) for Output Matrix

## Shape Error
import numpy as np
e_weights=[
[0.2, 0.8, -0.5, 1.0], 
[0.5, -0.91, 0.26, -0.5], 
[-0.26, -0.27, 0.17, 0.87]
]

e_inputs= [[1, 2, 3, 2.5],
           [2,5,-1,2],
           [-1.5, 2.7, 3.3, -0.8]]

biases=[
    2,3,0.5
]

## e_output=np.dot(e_inputs, e_weights)
# this will produce an error, because we have 4 values but only 3 arrays,
# so when the matrix product tries to multiply the 4th value of a row, we do not have a value to multiply with,
# because there is no 4th row to draw from


# (!) solution is to transpose
# (?) offene Frage für später: verfälscht es nicht die Daten einfach zu Transponieren??

output=np.dot(e_inputs,np.array(e_weights).T) + biases
# print("Layer 1 output: ", output)


### Stacking Layers
weights2 = [[0.1, -0.14, 0.5],
[-0.5, 0.12, -0.33],
[-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

# layer 1 "output" becomes input for Layer 2
layer2_output = np.dot(output, np.array(weights2).T) + biases2
# print(layer2_output)


### Programmatic approach to stacking layers (introducing hidden layers)
X = e_inputs # X is usually the name of inputs in Machine Learning

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) *0.1 #making sure number is small
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
## start by initializing values
# if nothing saved / already available
# start with rand's -1 to 1, the tighter the better
# normalize and scale dataset (meaning stays the same, while values become scmaller)
# print(np.random.randn(4,3))

## Actually pass shape 
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2) # input needs to be size of prev output layer

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)