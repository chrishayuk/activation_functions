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

def leaky_relu(x, alpha=0.01):
    # Leaky ReLU: allows small negative slope
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    # Exponential Linear Unit
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def prelu(x, alpha=0.25):
    # Parametric ReLU: learnable negative slope (alpha is fixed here for demo)
    return np.where(x > 0, x, alpha * x)

def mish(x):
    # Mish: x * tanh(softplus(x))
    return x * np.tanh(np.log(1 + np.exp(x)))

def geglu(x):
    # GeGLU: x * gelu(x) — GELU-gated linear unit
    return x * gelu(x)

def telu(x):
    # TeLU: x * tanh(exp(x))
    return x * np.tanh(np.exp(x))
