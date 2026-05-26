### What are activation functions?
# with only linear approaches, we could never fit the data to non-linear distributions (e.g. sine-distributions)

## Step-Function
# if input is greater than zero, then output is 1, otherwise 0
# usually output function has different step function than hidden layers

## Sigmoid function
# useful to optimize / decreae loss
# what is impact of weights / biases on values?
# > if X greater than zero, than Output is X, otherwise 0
# but has issue with "Vanishing gradient problem"

## Rectified Linear Function
# eigentlich nur "Geraden-Abschnitte" mit Start und Endpunkt, die einen nicht-linearen Shape annähernd ergeben können
# measuring when the input "activates" or "deactivates"
# the bias can shift the "activation point" around and thus create non-linear "graphs"
# not really "linear", clipping to zero makes
# https://youtu.be/gmjzbpSVY1A?si=iLdF4Zl8c3O1bwIT&t=639

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2,5,-1,2],
    [-1.5, 2.7, 3.3, -0.8]]

inputs=[0,2,-1,3.3,-2.7,1.1,2.2,-100]
output=[]

for i in inputs:
    # if i > 0:
    #     output.append(i)
    # else: output.append(0)
    ## prettier:
    output.append(max(0,i))
    
# print(output)
    
## Proper implementation
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) *0.1 #making sure number is small
        self.biases = np.zeros((1,n_neurons)) # erstellt array mit 0's
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)
        

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)



## function to create spiral example data
#https://cs231n.github.io/neural-networks-case-study/
def create_data(points, classes):
    X = np. zeros ((points*classes, 2))
    y = np.zeros (points*classes, dtype='uint8')

    for class_number in range(classes) :
        ix = range(points*class_number, points*(class_number+1))
        r = np. linspace(0.0, 1, points)
        # radius
        t = np. linspace(class_number*4, (class_number+1) *4, points) + np. random. rand(points) *0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y
    
X, y = create_data(100,3)

# Amping down the input because only 2 features in generated Data (x,y Data)
layer1 = Layer_Dense(2,5)
# takes all values from the neurons and produce activation for entire layer1 
activation1 = Activation_ReLu()

layer1.forward(X)
# this will have negative values
print(layer1.output)

# will actually run the activation function
activation1.forward(layer1.output)
print(activation1.output)