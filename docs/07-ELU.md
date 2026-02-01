# ELU
Exponential Linear Unit

ELU uses an exponential function for negative inputs, producing smooth outputs that approach -α asymptotically. This pushes the mean activation closer to zero, acting as a form of built-in normalization that can speed up training.

## Equation
ELU(x) = x if x > 0, else α · (exp(x) - 1)

where α controls the saturation value for negative inputs (default 1.0).

## Pros
- Zero-Centered Outputs: Negative outputs push the mean activation closer to zero, which can improve gradient dynamics.
- Smooth for Negative Inputs: The exponential curve provides a smooth gradient for negative values, unlike the sharp cutoff in ReLU.
- Noise Robustness: The saturation for large negative values makes ELU more robust to noise.

## Drawback
More computationally expensive than ReLU due to the exponential calculation for negative inputs. The saturation at -α can also slow down learning for very negative activations.

## Typical Use Cases
Used in deeper networks where zero-centered activations improve convergence. Popular in autoencoders and generative models where smooth gradients are beneficial.

## Plot Analysis
ELU is linear and identical to ReLU for positive inputs. For negative inputs, instead of being flat at zero (ReLU) or slightly negative (Leaky ReLU), ELU curves smoothly downward toward -α. This exponential saturation creates a soft floor that prevents extreme negative activations while still allowing meaningful negative gradients near zero.
