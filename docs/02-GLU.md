# GLu
Gated Linear Unit
GLU uses a gating mechanism (multiplying input by the sigmoid of the same input), enabling the model to selectively pass information.

## Equation
GLU(x)=(x⋅σ(x))

## Pro's
- Improved Gradient Flow: Helps in mitigating issues like gradient vanishing, especially in RNNs and transformers.

## Drawback
More Parameters: Typically, the gate requires extra parameters, which can increase computational costs.

## Typical Use Cases
NLP and transformer-based architectures (e.g., transformers, RNNs) where more control over gradient flow and selective information passage is beneficial.

## Plot Analysis
GLU introduces a gating mechanism by multiplying the input by its sigmoid transformation, which scales inputs based on their likelihood of being positive.

It gently suppresses negative values rather than cutting them off completely, resulting in a smoother curve around zero compared to ReLU.

This smoother suppression makes it useful in transformers and other architectures where gradient flow needs to be preserved. You can see that GLU hovers just below zero for negative inputs and smoothly transitions upward.