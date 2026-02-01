import numpy as np
import pandas as pd
from activation_functions import (
    relu, glu, swiglu, gelu, silu,
    leaky_relu, elu, prelu, mish, geglu, telu
)

def generate_dataframe():
    # Generate input values from -5 to 5
    x = np.linspace(-5, 5, 100)

    # Creating a table of values for each activation function
    data = {
        "Input": x,
        "ReLU": relu(x),
        "Leaky ReLU": leaky_relu(x),
        "ELU": elu(x),
        "PReLU": prelu(x),
        "GELU": gelu(x),
        "SiLU": silu(x),
        "GLU": glu(x),
        "SwiGLU": swiglu(x),
        "GeGLU": geglu(x),
        "Mish": mish(x),
        "TeLU": telu(x),
    }

    # Convert to DataFrame for better readability
    activation_df = pd.DataFrame(data)

    # return the dataframe
    return activation_df