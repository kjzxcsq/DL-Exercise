import numpy as np
import copy
from scipy.signal import correlate, convolve
from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """
        Initializes the Conv layer.

        Args:
            stride_shape (tuple or int): The stride of the convolution.
            convolution_shape (tuple): Shape of the kernel/filter.
            num_kernels (int): Number of kernels.
        """
        super().__init__()
        # Determine convolution dimensionality
        if len(convolution_shape) == 2:
            # 1D Convolution: [channels, kernel_length]
            self.dim = 1
        elif len(convolution_shape) == 3:
            # 2D Convolution: [channels, kernel_height, kernel_width]
            self.dim = 2
        else:
            raise ValueError("convolution_shape must be either 2D or 3D tuple")

        # Set stride_shape
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,) * self.dim
        else:
            if len(stride_shape) != self.dim:
                raise ValueError(f"stride_shape must have length {self.dim}")
            self.stride_shape = stride_shape

        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.padded_input = None

        # Initialize weights and bias uniformly in [0, 1)
        self.weights = np.random.uniform(0, 1, (num_kernels,) + convolution_shape)
        self.bias = np.random.uniform(0, 1, num_kernels)

        # Gradient placeholders
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        # Optimizers
        self.optimizer_weights = None
        self.optimizer_bias = None

    def forward(self, input_tensor):
        """
        Forward pass for the convolutional layer.

        Args:
            input_tensor (np.array): Input tensor with shape (batch, channels, spatial_dim...).

        Returns:
            np.array: Output tensor after applying convolution.
        """
        self.input_tensor = input_tensor
        batch_size, in_channels = input_tensor.shape[:2]
        spatial_dims = input_tensor.shape[2:]

        # Determine padding for 'same' convolution
        pad_width = []
        for i in range(self.dim):
            kernel_size = self.convolution_shape[i+1]
            pad_width.append(((kernel_size) // 2, (kernel_size - 1) // 2))

        # Pad the input tensor
        if self.dim == 1:
            self.padded_input = np.pad(input_tensor, ((0,0), (0,0), (pad_width[0][0], pad_width[0][1])), mode='constant')
        else:
            self.padded_input = np.pad(input_tensor, ((0,0), (0,0)) + tuple(pad_width), mode='constant')

        # Calculate output spatial dimensions
        output_shape = []
        for i in range(self.dim):
            out_dim = (spatial_dims[i] + pad_width[i][0] + pad_width[i][1] - self.convolution_shape[i+1]) // self.stride_shape[i] + 1
            output_shape.append(out_dim)

        # Initialize output tensor
        output_tensor = np.zeros((batch_size, self.num_kernels) + tuple(output_shape))

        # Perform convolution
        for b in range(batch_size):
            for k in range(self.num_kernels):
                conv_sum = None
                for c in range(in_channels):
                    if self.dim == 1:
                        # 1D Convolution: correlate along the single spatial dimension
                        conv = correlate(self.padded_input[b, c], self.weights[k, c], mode='valid')
                        # Apply stride
                        conv = conv[::self.stride_shape[0]]
                    else:
                        # 2D Convolution: correlate along both spatial dimensions
                        conv = correlate(self.padded_input[b, c], self.weights[k, c], mode='valid')
                        # Apply stride
                        conv = conv[::self.stride_shape[0], ::self.stride_shape[1]]
                    if conv_sum is None:
                        conv_sum = conv
                    else:
                        conv_sum += conv
                # Add bias
                conv_sum += self.bias[k]
                output_tensor[b, k] = conv_sum

        return output_tensor

    def backward(self, error_tensor):
        """
        Backward pass for the convolutional layer.

        Args:
            error_tensor (np.array): Error tensor from the next layer.

        Returns:
            np.array: Error tensor for the previous layer.
        """
        batch_size, in_channels = self.input_tensor.shape[:2]
        spatial_dims_input = self.input_tensor.shape[2:]
        spatial_dims_error = error_tensor.shape[2:]

        # Initialize gradients
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        grad_input = np.zeros_like(self.input_tensor)

        # Upsample the error tensor for stride > 1
        upsampled_error = np.zeros((batch_size, self.num_kernels) + tuple(spatial_dims_input))
        if self.dim == 1:
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    upsampled_error[b, k, ::self.stride_shape[0]] = error_tensor[b, k]
        else:
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    upsampled_error[b, k, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b, k]

        # Compute gradients
        for b in range(batch_size):
            for c in range(in_channels):
                # Stack kernels across output channels for current input channel
                stacked_kernels = np.flip(np.stack([self.weights[k, c] for k in range(self.num_kernels)], axis=0), axis=0)
                grad = convolve(upsampled_error[b], stacked_kernels, mode='same')
                grad_input[b, c] = grad[(len(grad) - 1) // 2]

            for k in range(self.num_kernels):
                # Gradient w.r.t. bias
                self._gradient_bias[k] += np.sum(upsampled_error[b, k])

                for c in range(in_channels):
                    # Gradient w.r.t. weights
                    input_segment = self.padded_input[b, c]
                    self._gradient_weights[k, c] += correlate(input_segment, upsampled_error[b, k], mode='valid')


        # Update weights and biases using optimizer
        if self.optimizer_weights:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self.optimizer_bias:
            self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return grad_input

    def initialize(self, weights_initializer, bias_initializer):
        """
        Reinitialize weights and biases using initializers.

        Args:
            weights_initializer: Weights initializer object.
            bias_initializer: Bias initializer object.
        """
        # Calculate fan_in and fan_out correctly
        if self.dim == 1:
            fan_in = self.convolution_shape[0] * self.convolution_shape[1]
            fan_out = self.num_kernels * self.convolution_shape[1]
        else:
            fan_in = self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2]
            fan_out = self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2]

        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer_weights = copy.deepcopy(optimizer)
        self.optimizer_bias = copy.deepcopy(optimizer)
