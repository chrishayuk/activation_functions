# GeGLU
GELU-Gated Linear Unit

GeGLU is a gated activation function that uses GELU as the gating mechanism instead of Swish (as in SwiGLU) or sigmoid (as in GLU). It was introduced alongside other GLU variants by Noam Shazeer in 2020 and is notably used in Google's Gemma model family.

## Equation
GeGLU(x) = x · GELU(x)

In practice within a transformer FFN, this operates on split projections:
GeGLU(x, W, V) = GELU(xW) · (xV)

## Pros
- GELU-Based Gating: Leverages the probabilistic gating properties of GELU, providing smooth and nuanced information control.
- Strong Performance in Encoder Models: Follows naturally from encoder-style models that already use GELU (BERT, ViT).
- Improved Over Standard FFN: Like SwiGLU, the gated architecture outperforms the traditional expand-activate-contract FFN pattern.

## Drawback
Requires three weight matrices in the FFN instead of two (same trade-off as SwiGLU). The inner dimension is typically reduced to keep parameter count equivalent, but the computation is still more involved.

## Typical Use Cases
Used in Google's Gemma models. Well-suited for encoder-style transformers and models that build on the GELU/BERT tradition. Can be used anywhere SwiGLU is used as an alternative gating strategy.

## Plot Analysis
GeGLU's curve is similar to SwiGLU but with GELU's characteristic probabilistic gating shape. For positive inputs, GeGLU grows faster than SwiGLU because GELU approaches 1 more quickly than SiLU does. For negative inputs, GeGLU produces a small positive bump (similar to SwiGLU) before settling toward zero, reflecting the multiplicative interaction between x and GELU(x).
