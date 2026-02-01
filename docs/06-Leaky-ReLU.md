# Leaky ReLU
Leaky Rectified Linear Unit

Leaky ReLU addresses the "dying ReLU" problem by allowing a small, non-zero gradient for negative inputs. Instead of outputting zero for all negative values, it multiplies them by a small constant (typically 0.01), keeping neurons alive during training.

## Equation
LeakyReLU(x) = x if x > 0, else α · x

where α is a small constant (default 0.01).

## Pros
- Prevents Dying Neurons: The small negative slope ensures that neurons always have a non-zero gradient, avoiding the "dying ReLU" problem.
- Simple and Fast: Nearly as computationally cheap as standard ReLU.
- Better Gradient Flow: Negative inputs still contribute to learning, improving convergence in some architectures.

## Drawback
The fixed negative slope (α) is a hyperparameter that may not be optimal for all tasks. The small negative values can also introduce noise in some cases.

## Typical Use Cases
Used as a drop-in replacement for ReLU in CNNs and fully connected networks where dying neurons are a concern. Common in GANs (particularly the discriminator) where gradient flow through negative values is important.

## Plot Analysis
Leaky ReLU looks almost identical to ReLU for positive inputs — a straight line with slope 1. The key difference is in the negative range: instead of being flat at zero, there is a slight downward slope (barely visible at α=0.01). This small but non-zero gradient for negative inputs is what prevents neurons from permanently dying during training.
