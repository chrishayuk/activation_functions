# ReLU
Rectified Linear Unit

## Equation
ReLU(x)=max(0,x)

## Pro's
- Simplicity: It’s computationally simple and fast, making it popular in many neural networks.
- Non-linearity: Provides non-linearity, helping the model learn complex patterns.

## Drawback
Can lead to the “dying ReLU” problem where neurons output zero for all inputs due to negative gradients in training, especially in deeper networks.

## Typical Use Cases
Typically used in CNN's as good with Convolutional Layers or fully connected layers.

## Plot Analysis
ReLU outputs zero for negative inputs, then increases linearly for positive inputs.
This hard threshold at zero makes it simple but can cause some neurons to "die" (output zero) permanently during training, especially for large negative values.
In this plot, you can see it stays at zero for all negative inputs, then sharply starts to increase as the input becomes positive.