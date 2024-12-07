import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    """
    A max-pooling layer that reduces the spatial dimensions of the input tensor
    by taking the maximum value within non-overlapping windows defined by the
    pooling shape and stride.

    Attributes:
        stride_shape (tuple): The stride (vertical, horizontal) by which the 
            pooling window is moved across the input.
        pooling_shape (tuple): The size (height, width) of the pooling window.
        input_shape (tuple): The shape of the input tensor stored during the
            forward pass, used to correctly shape the gradients during backward pass.
        max_positions (np.ndarray): The relative positions of the maximum values 
            within each pooling window, stored for use in the backward pass.
    """

    def __init__(self, stride_shape, pooling_shape):
        """
        Initialize the Pooling layer.

        Args:
            stride_shape (tuple or int): The stride of the pooling window. If an integer
                is provided, it will be used for both height and width. If a tuple is provided, 
                it should contain (stride_height, stride_width).
            pooling_shape (tuple): The size (pool_height, pool_width) of the pooling window.
        """
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.max_positions = None

    def forward(self, input_tensor):
        """
        Perform the forward pass of the max-pooling layer.

        This method scans the input tensor with a window of size `pooling_shape` 
        and stride `stride_shape`, and takes the maximum value within each window.

        Args:
            input_tensor (np.ndarray): A 4D input tensor with shape 
                (batch_size, channels, height, width).

        Returns:
            np.ndarray: The pooled output tensor with reduced height and width dimensions. 
            The shape will be (batch_size, channels, out_height, out_width), where:
            out_height = (height - pool_height) // stride_height + 1
            out_width = (width - pool_width) // stride_width + 1
        """
        self.input_shape = input_tensor.shape
        batch_size, channels, input_height, input_width = self.input_shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        # Compute output spatial dimensions
        out_height = (input_height - pool_height) // stride_height + 1
        out_width = (input_width - pool_width) // stride_width + 1

        # Initialize output tensor
        output_tensor = np.zeros((batch_size, channels, out_height, out_width))

        # Store positions of max elements to use in backward pass
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
                        self.max_positions[b, c, h, w] = max_position

        return output_tensor

    def backward(self, error_tensor):
        """
        Perform the backward pass of the max-pooling layer.

        This method propagates the error back to the input tensor. Only the 
        positions that were chosen as maximums during the forward pass receive 
        the gradient from the error tensor.

        Args:
            error_tensor (np.ndarray): The gradient tensor from the next layer, 
                with shape (batch_size, channels, out_height, out_width).

        Returns:
            np.ndarray: The gradient tensor with the same shape as the input 
            (batch_size, channels, input_height, input_width), where only the positions 
            of the maximum values in each pooling window receive the incoming gradient.
        """
        batch_size, channels, out_height, out_width = error_tensor.shape
        _, _, input_height, input_width = self.input_shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        # Initialize gradient w.r.t. input
        grad_input = np.zeros((batch_size, channels, input_height, input_width))

        # Distribute gradients to the positions of the maximum values
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        start_h = h * stride_height
                        start_w = w * stride_width
                        end_h = start_h + pool_height
                        end_w = start_w + pool_width

                        max_position = self.max_positions[b, c, h, w]
                        max_h, max_w = max_position
                        grad_input[b, c, start_h:end_h, start_w:end_w][max_h, max_w] += error_tensor[b, c, h, w]

        return grad_input