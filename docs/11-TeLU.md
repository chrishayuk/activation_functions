# TeLU
Hyperbolic Tangent Exponential Linear Unit

TeLU is one of the newest activation functions, proposed in December 2024 by researchers at the University of South Florida. It is designed as a smooth, drop-in replacement for ReLU that combines the exponential and hyperbolic tangent functions to produce a bounded gating effect with strong gradient properties.

## Equation
TeLU(x) = x · tanh(exp(x))

## Pros
- Drop-in ReLU Replacement: Can replace ReLU in existing architectures without structural changes.
- Smooth and Analytic: Infinitely differentiable, providing stable gradients throughout training.
- Strong Benchmarks: Claims to outperform GELU, SiLU, ELU, and Mish across CNNs (ResNet on ImageNet), transformers (Text8), and RNNs (Penn Treebank).
- Fast Convergence: The exponential gating drives tanh to saturation quickly for positive inputs, producing near-identity behavior that aids gradient flow.

## Drawback
Very new (2024) and not yet widely adopted or battle-tested at scale. The exp(x) term can cause numerical overflow for large positive inputs, requiring careful implementation with clamping.

## Typical Use Cases
Proposed as a general-purpose activation for any architecture currently using ReLU, GELU, or SiLU. Benchmarked on image classification (ResNet18/34), language modeling (transformers), and sequence modeling (RNNs). Adoption in production LLMs or large-scale vision models remains to be seen.

## Plot Analysis
TeLU behaves similarly to SiLU for moderate inputs but with a sharper transition. For positive inputs, tanh(exp(x)) saturates to 1 very quickly, making TeLU approach the identity function (y = x) faster than SiLU or GELU. For negative inputs, exp(x) approaches 0 and tanh(0) = 0, so TeLU smoothly suppresses negative values toward zero — similar to GELU but with a slightly different curvature. The result is a function that is "more ReLU-like" than SiLU or GELU while retaining smoothness.
