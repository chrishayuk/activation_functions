import numpy as np

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def glu(x):
     # GLU uses sigmoid as a gate
    return x * (1 / (1 + np.exp(-x))) 

def swiglu(x):
    # Swish(x) = x * sigmoid(x)
    swish = x / (1 + np.exp(-x))  
    return x * swish

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
