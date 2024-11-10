import numpy as np
import pandas as pd
from activation_functions import relu, glu, swiglu, gelu

def generate_dataframe():
    # Generate input values from -5 to 5
    x = np.linspace(-5, 5, 100)

    # Creating a table of values for each activation function
    data = {
        "Input": x,
        "ReLU Output": relu(x),
        "GLU Output": glu(x),
        "Swiglu Output": swiglu(x),
        "GeLU Output": gelu(x)
    }

    # Convert to DataFrame for better readability
    activation_df = pd.DataFrame(data)

    # return the dataframe
    return activation_df