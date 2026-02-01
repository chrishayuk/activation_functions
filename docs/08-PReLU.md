# PReLU
Parametric Rectified Linear Unit

PReLU generalizes Leaky ReLU by making the negative slope a learnable parameter. During training, the network learns the optimal slope for negative inputs rather than using a fixed value, allowing each layer (or even each channel) to adapt its activation behavior.

## Equation
PReLU(x) = x if x > 0, else α · x

where α is a learnable parameter (initialized to 0.25 in many implementations).

## Pros
- Adaptive Negative Slope: The learned α can be different per layer or per channel, giving the network more flexibility than fixed Leaky ReLU.
- No Dying Neurons: Like Leaky ReLU, negative inputs always produce non-zero gradients.
- Minimal Extra Parameters: Only one extra parameter per channel/layer, negligible overhead.

## Drawback
The learnable parameter adds a small amount of complexity to the training process. On small datasets, the extra parameter can lead to slight overfitting compared to fixed alternatives.

## Typical Use Cases
Commonly used in image recognition networks (e.g., early ResNets, VGG variants) where learning the optimal negative slope per layer can improve accuracy. Also useful in any deep architecture where Leaky ReLU helps but the fixed slope feels limiting.

## Plot Analysis
PReLU looks identical to Leaky ReLU in shape — linear for positive inputs, with a straight negative slope for negative inputs. The difference is that the negative slope (shown here at α=0.25 for visibility) is learned during training. At α=0.25, the negative slope is more visible than Leaky ReLU's default 0.01, creating a steeper decline for negative inputs.
