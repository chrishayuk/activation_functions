import numpy as np

# Define the Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the Swish function using Sigmoid
def swish(x):
    # sigmoid as a gate
    return x * sigmoid(x)

# Define activation functions using Sigmoid and Swish
def relu(x):
    return np.maximum(0, x)

def glu(x):
    # GLU applies an element-wise gate
    return x * sigmoid(x)

def gelu(x):
    # gaussian
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    # SiLU (Swish): x * sigmoid(x)
    return swish(x)

def swiglu(x):
    # SwiGLU uses Swish as a gate
    return x * swish(x)




