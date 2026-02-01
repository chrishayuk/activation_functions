# Activation Functions

A reference implementation and comparison of neural network activation functions, from classic to cutting-edge.

## Activation Functions Covered

| # | Function | Year | Used In |
|---|----------|------|---------|
| 1 | **ReLU** | 2010 | CNNs, fully connected layers |
| 2 | **Leaky ReLU** | 2013 | GANs, deep CNNs |
| 3 | **ELU** | 2015 | Autoencoders, deep networks |
| 4 | **PReLU** | 2015 | ResNets, image classification |
| 5 | **GLU** | 2016 | NLP, transformers |
| 6 | **GELU** | 2016 | BERT, GPT-2/3, Vision Transformers |
| 7 | **SiLU (Swish)** | 2017 | EfficientNet, LLaMA, YOLO |
| 8 | **Mish** | 2019 | YOLOv4/v5 |
| 9 | **SwiGLU** | 2020 | LLaMA, Mistral, PaLM, DeepSeek |
| 10 | **GeGLU** | 2020 | Gemma |
| 11 | **TeLU** | 2024 | Emerging — ResNet, transformers, RNNs |

## Project Structure

```
activation_functions.py      # All function implementations
show_plot.py                 # Side-by-side visualization (classic vs modern)
show_table.py                # Print comparison table
dataframe_utils.py           # Generate pandas DataFrame of outputs
test_activation_functions.py # Unit tests
docs/                        # Per-function documentation
```

## Getting Started

### Requirements

```
numpy
pandas
matplotlib
pytest
```

### Install dependencies

```bash
uv pip install numpy pandas matplotlib pytest
```

### Visualize all functions

```bash
uv run python show_plot.py
```

### Print comparison table

```bash
uv run python show_table.py
```

### Run tests

```bash
uv run pytest test_activation_functions.py -v
```

## Quick Reference

**Classic (ReLU family):** Simple thresholding with variations on how negative inputs are handled.
- ReLU: hard zero for negatives
- Leaky ReLU: small fixed slope for negatives (0.01)
- PReLU: learned slope for negatives
- ELU: exponential curve for negatives

**Smooth (drop-in replacements):** Smooth approximations that allow small negative signals through.
- GELU: probabilistic gating via Gaussian CDF — standard in encoder transformers (BERT, ViT)
- SiLU/Swish: x * sigmoid(x) — standard in decoder LLMs (LLaMA, Mistral)
- Mish: x * tanh(softplus(x)) — used in YOLO
- TeLU: x * tanh(exp(x)) — newest contender (2024)

**Gated (FFN replacements):** Replace the standard transformer FFN with a gated architecture using 3 weight matrices.
- GLU: sigmoid gating
- SwiGLU: SiLU/Swish gating — dominant in modern LLMs
- GeGLU: GELU gating — used in Gemma
