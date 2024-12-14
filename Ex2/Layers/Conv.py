import numpy as np
import copy
from scipy.signal import correlate, convolve
from .Base import BaseLayer

class Conv(BaseLayer):
    """
    A convolutional layer that applies one-dimensional or two-dimensional convolutions 
    on the input tensor.

    This layer supports both 1D and 2D convolutions. For 1D convolutions, the input 
    shape is expected to be (batch_size, channels, width), and for 2D convolutions, 
    the input shape should be (batch_size, channels, height, width).

    Attributes:
        stride_shape (tuple): The stride for the convolution. It can be a single integer 
            (for 1D) or a tuple of two integers (for 2D).
        convolution_shape (tuple): The shape of the convolution kernels. For 1D, 
            (in_channels, kernel_width), and for 2D, (in_channels, kernel_height, kernel_width).
        num_kernels (int): The number of kernels (output channels) in this convolutional layer.
        trainable (bool): Indicates if the layer has trainable parameters.
        padded_input (np.ndarray or None): The input tensor after padding, stored during forward pass.
        weights (np.ndarray): The weights of the layer with shape 
            (num_kernels, in_channels, [kernel_height,] kernel_width).
        bias (np.ndarray): The biases for each kernel, of shape (num_kernels,).
        _gradient_weights (np.ndarray): The gradient of the loss with respect to the weights.
        _gradient_bias (np.ndarray): The gradient of the loss with respect to the biases.
        optimizer_weights: The optimizer instance for the weights.
        optimizer_bias: The optimizer instance for the biases.
        dim (int): The dimensionality of the convolution (1 or 2).
    """

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """
        Initialize the Conv layer.

        This method determines if the convolution is 1D or 2D based on the 
        shape of `convolution_shape`, sets strides, and initializes weights 
        and biases uniformly in the range [0, 1).

        Args:
            stride_shape (int or tuple): The stride of the convolution. If an integer is 
                provided for a 1D convolution, it will be converted to a tuple. For a 2D 
                convolution, a tuple of two integers should be provided.
            convolution_shape (tuple): Shape of the kernel/filter. For 1D: (in_channels, kernel_width).
                For 2D: (in_channels, kernel_height, kernel_width).
            num_kernels (int): The number of kernels (output channels).
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
        Perform the forward pass of the convolutional layer.

        This method applies convolution to the input tensor using the stored weights 
        and biases. The input is padded to achieve 'same' convolution behavior (output 
        size matches input size when stride=1).

        Args:
            input_tensor (np.ndarray): The input tensor of shape:
                - (batch_size, in_channels, width) for 1D convolution.
                - (batch_size, in_channels, height, width) for 2D convolution.

        Returns:
            np.ndarray: The output tensor after applying convolution, with shape:
                - (batch_size, num_kernels, new_width) for 1D convolution.
                - (batch_size, num_kernels, new_height, new_width) for 2D convolution.
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
                        # conv = correlate(self.input_tensor[b, c], self.weights[k, c], mode='same')
                        # Apply stride
                        conv = conv[::self.stride_shape[0]]
                    else:
                        # 2D Convolution: correlate along both spatial dimensions
                        conv = correlate(self.padded_input[b, c], self.weights[k, c], mode='valid')
                        # conv = correlate(self.input_tensor[b, c], self.weights[k, c], mode='same')
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
        Perform the backward pass of the convolutional layer.

        This method computes the gradient of the loss with respect to the inputs, weights, 
        and biases using the provided error tensor from the subsequent layer. It also applies 
        upsampling to the error tensor if strides are greater than one and updates the 
        parameters if optimizers are set.

        Args:
            error_tensor (np.ndarray): The error tensor from the next layer, with shape:
                - (batch_size, num_kernels, width) for 1D convolution.
                - (batch_size, num_kernels, height, width) for 2D convolution.

        Returns:
            np.ndarray: The error tensor to pass to the previous layer, shaped the same 
            as the input tensor of this layer.
        """
        batch_size, in_channels = self.input_tensor.shape[:2]
        spatial_dims_input = self.input_tensor.shape[2:]

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
                # Extract the central slice (for correct alignment)
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
        Reinitialize weights and biases using the provided initializers.

        This method calculates the fan_in and fan_out based on the convolution shape 
        and then uses the given initializers to reinitialize the weights and biases.

        Args:
            weights_initializer: An object with an `initialize` method for weights.
            bias_initializer: An object with an `initialize` method for biases.
        """
        # Calculate fan_in and fan_out
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
        """
        np.ndarray: The gradient of the loss with respect to the weights.
        """
        return self._gradient_weights

    @property
    def gradient_bias(self):
        """
        np.ndarray: The gradient of the loss with respect to the biases.
        """
        return self._gradient_bias

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        """
        Returns the current optimizer. Setting this property 
        also sets the optimizers for weights and biases.
        """
        return self.optimizer_weights, self.optimizer_bias

    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer_weights = copy.deepcopy(optimizer)
        self.optimizer_bias = copy.deepcopy(optimizer)
