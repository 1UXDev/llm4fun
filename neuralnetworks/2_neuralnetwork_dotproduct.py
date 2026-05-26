### Shape:
# a1_weights = [0.2, 0.8, -0.5] > Shape=(4)
# weights=[[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]] > Shape=(2,4) > Matrix

# Shape = (3,2,4) > 3d Array, first level 3 items, each has 2 inside, of which each has 4 items
# w=[
# [[a,b,a2,b2],[c,d,c2,d2]],
# [[e,f,e2,f2], [g,h,g2,h2]],
# [[i,j,i2,j2],[k,l,k2,l2]]
# ]

# Tensor > in Deeplearning an array (but outside of deeplearning probably more)

### Dot Product
# .dot() basically does what _intro did manually
import numpy as np

inputs = [1, 2, 3, 2.5]
weights=[
[0.2, 0.8, -0.5, 1.0], 
[0.5, -0.91, 0.26, -0.5], 
[-0.26, -0.27, 0.17, 0.87]
]
biases=[
    2,3,0.5
]

# weights MUST come first
outputs=np.dot(weights, inputs) + biases
print(outputs)