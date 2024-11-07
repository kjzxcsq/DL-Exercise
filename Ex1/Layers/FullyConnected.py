import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    """
    Fully Connected layer that performs a linear operation on its input.
    
    Attributes:
        trainable (bool): Indicates if the layer has trainable parameters (always True for this layer).
        weights (np.ndarray): The weights of the layer, initialized uniformly.
        _optimizer: The optimizer instance used for updating weights.
        _gradient_weights (np.ndarray): The gradient with respect to the weights.
    """

    def __init__(self, input_size, output_size):
        """
        Initialize the FullyConnected layer with input and output sizes.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
        """
        super().__init__()
        self.trainable = True
        # Initialize weights uniformly in the range [0, 1)
        self.weights = np.random.rand(input_size + 1, output_size)
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        """
        Perform the forward pass of the layer.

        Args:
            input_tensor (np.ndarray): The input tensor (batch_size x input_size).

        Returns:
            np.ndarray: The output tensor for the next layer (batch_size x output_size).
        """
        batch_size = input_tensor.shape[0]
        # Add a bias term of ones to the input tensor
        self.input_tensor = np.hstack([input_tensor, np.ones((batch_size, 1))])
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        """
        Perform the backward pass of the layer.

        Args:
            error_tensor (np.ndarray): The error tensor from the next layer (batch_size x output_size).

        Returns:
            np.ndarray: The error tensor for the previous layer (batch_size x input_size).
        """
        # Calculate the gradient with respect to the weights
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # Compute the gradient for the input tensor
        gradient_input = np.dot(error_tensor, self.weights[:-1, :].T)

        # Update weights using the optimizer if it is set
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        
        return gradient_input

    @property
    def optimizer(self):
        """Get the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        """Set the optimizer."""
        self._optimizer = opt

    @property
    def gradient_weights(self):
        """Get the gradient with respect to the weights."""
        return self._gradient_weights