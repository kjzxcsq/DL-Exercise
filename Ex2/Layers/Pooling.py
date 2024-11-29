import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.max_positions = None  # Store max positions for backward

    def forward(self, input_tensor):
        """
        Forward pass for max-pooling.
        :param input_tensor: 4D tensor with shape (batch_size, channels, height, width).
        :return: Pooled output tensor.
        """
        self.input_shape = input_tensor.shape
        batch_size, channels, input_height, input_width = self.input_shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        # Calculate output dimensions
        out_height = (input_height - pool_height) // stride_height + 1
        out_width = (input_width - pool_width) // stride_width + 1

        # Initialize output tensor and store max positions
        output_tensor = np.zeros((batch_size, channels, out_height, out_width))
        self.max_positions = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        start_h = h * stride_height
                        start_w = w * stride_width
                        end_h = start_h + pool_height
                        end_w = start_w + pool_width

                        pool_region = input_tensor[b, c, start_h:end_h, start_w:end_w]
                        max_position = np.unravel_index(np.argmax(pool_region, axis=None), pool_region.shape)
                        output_tensor[b, c, h, w] = pool_region[max_position]
                        self.max_positions[b, c, h, w] = max_position  # Relative position within the pool region

        return output_tensor

    def backward(self, error_tensor):
        """
        Backward pass for max-pooling.
        :param error_tensor: 4D tensor with shape of the forward output tensor.
        :return: Gradient tensor with shape of the forward input tensor.
        """
        batch_size, channels, out_height, out_width = error_tensor.shape
        _, _, input_height, input_width = self.input_shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        # Initialize gradient tensor
        grad_input = np.zeros((batch_size, channels, input_height, input_width))

        # Distribute gradients
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        start_h = h * stride_height
                        start_w = w * stride_width
                        end_h = start_h + pool_height
                        end_w = start_w + pool_width

                        # Get the max position within the receptive field
                        max_position = self.max_positions[b, c, h, w]
                        max_h, max_w = max_position
                        grad_input[b, c, start_h:end_h, start_w:end_w][max_h, max_w] += error_tensor[b, c, h, w]

        return grad_input