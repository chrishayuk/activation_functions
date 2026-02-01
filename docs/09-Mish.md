# Mish
Mish Activation Function

Mish is a smooth, non-monotonic, self-regularizing activation function. It was proposed by Diganta Misra in 2019 and gained traction through its adoption in YOLOv4 and YOLOv5 object detection models. Its smooth curve and slight negative allowance provide properties similar to Swish but with a different mathematical formulation.

## Equation
Mish(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + exp(x)))

## Pros
- Smooth and Non-monotonic: The slight dip below zero for negative inputs allows information to flow even for negative activations.
- Self-Regularizing: The bounded negative region acts as a form of implicit regularization.
- Strong Empirical Results: Demonstrated improvements in object detection (YOLO) and image classification tasks.
- Unbounded Above: Like ReLU, positive outputs are not capped, avoiding saturation for positive inputs.

## Drawback
More computationally expensive than ReLU, SiLU, or GELU due to the combination of softplus, tanh, and multiplication. Being superseded by SiLU/Swish in newer architectures.

## Typical Use Cases
Primarily used in computer vision, especially object detection models (YOLOv4, YOLOv5). Also found in some CNN classification architectures where smooth activations improve convergence.

## Plot Analysis
Mish looks very similar to SiLU/Swish — both are smooth, both allow a small negative dip, and both grow linearly for large positive inputs. The subtle difference is in the negative region: Mish's dip is slightly shallower and smoother due to the tanh(softplus(x)) formulation. For large positive inputs, Mish and SiLU become nearly identical.
