import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    """
    Rectified Linear Unit (ReLU) activation function layer.
    
    Inherits from:
        BaseLayer: The base class for all layers in the framework.
    """

    def __init__(self):
        """
        Initialize the ReLU layer. This layer does not have trainable parameters.
        """
        super().__init__()

    def forward(self, input_tensor):
        """
        Perform the forward pass of the ReLU activation.

        Args:
            input_tensor (np.ndarray): The input tensor.

        Returns:
            np.ndarray: The output tensor where all negative values are set to zero.
        """
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        """
        Perform the backward pass of the ReLU activation.

        Args:
            error_tensor (np.ndarray): The error tensor from the next layer.

        Returns:
            np.ndarray: The error tensor for the previous layer, 
                        where gradients are zero for non-positive inputs.
        """
        return error_tensor * (self.input_tensor > 0)
