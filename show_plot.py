import numpy as np
import matplotlib.pyplot as plt
from activation_functions import relu, glu, swiglu, gelu

# Generate input values from -5 to 5
x = np.linspace(-5, 5, 100)

# Set up a single figure for the plot
plt.figure(figsize=(10, 6))

# Plot the values
plt.plot(x, relu(x), label="ReLU", linestyle="--", color="blue", linewidth=2, marker='o', markevery=10)
plt.plot(x, glu(x), label="GLU", linestyle="-", color="orange", linewidth=2, marker='s', markevery=10)
plt.plot(x, swiglu(x), label="Swiglu", linestyle="-.", color="green", linewidth=2, marker='^', markevery=10)
plt.plot(x, gelu(x), label="GELU", linestyle=":", color="red", linewidth=2, marker='x', markevery=10)

# Set y-axis limits to focus between -1 and 5
plt.ylim(-1, 5)

# Set up title, labels, and legend
plt.title("Activation Functions: ReLU vs GLU vs Swiglu vs GELU")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(visible=True, linestyle=':', linewidth=0.7)

# Show the plot
plt.show()
