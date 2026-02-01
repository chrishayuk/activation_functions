import numpy as np
import matplotlib.pyplot as plt
from activation_functions import (
    relu, glu, swiglu, gelu, silu,
    leaky_relu, elu, prelu, mish, geglu, telu
)

# Generate input values from -5 to 5
x = np.linspace(-5, 5, 100)

# --- Plot 1: Classic Activation Functions ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ax1 = axes[0]
ax1.plot(x, relu(x), label="ReLU", linestyle="--", color="blue", linewidth=2, marker='o', markevery=10)
ax1.plot(x, leaky_relu(x), label="Leaky ReLU", linestyle="-", color="cyan", linewidth=2, marker='d', markevery=10)
ax1.plot(x, elu(x), label="ELU", linestyle="-.", color="purple", linewidth=2, marker='v', markevery=10)
ax1.plot(x, prelu(x), label="PReLU (a=0.25)", linestyle=":", color="brown", linewidth=2, marker='P', markevery=10)
ax1.plot(x, gelu(x), label="GELU", linestyle="-", color="red", linewidth=2, marker='x', markevery=10)
ax1.plot(x, silu(x), label="SiLU (Swish)", linestyle="--", color="green", linewidth=2, marker='^', markevery=10)
ax1.set_ylim(-2, 5)
ax1.set_title("Classic Activation Functions")
ax1.set_xlabel("Input")
ax1.set_ylabel("Output")
ax1.legend(loc="upper left")
ax1.grid(visible=True, linestyle=':', linewidth=0.7)

# --- Plot 2: Modern / Gated Activation Functions ---
ax2 = axes[1]
ax2.plot(x, glu(x), label="GLU", linestyle="-", color="orange", linewidth=2, marker='s', markevery=10)
ax2.plot(x, swiglu(x), label="SwiGLU", linestyle="-.", color="green", linewidth=2, marker='^', markevery=10)
ax2.plot(x, geglu(x), label="GeGLU", linestyle="--", color="magenta", linewidth=2, marker='D', markevery=10)
ax2.plot(x, mish(x), label="Mish", linestyle=":", color="teal", linewidth=2, marker='p', markevery=10)
ax2.plot(x, telu(x), label="TeLU", linestyle="-", color="crimson", linewidth=2, marker='*', markevery=10)
ax2.set_ylim(-2, 5)
ax2.set_title("Modern / Gated Activation Functions")
ax2.set_xlabel("Input")
ax2.set_ylabel("Output")
ax2.legend(loc="upper left")
ax2.grid(visible=True, linestyle=':', linewidth=0.7)

plt.tight_layout()
plt.show()
