import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    """
    SoftMax activation function layer used to transform logits into a probability distribution.
    
    Inherits from:
        BaseLayer: The base class for all layers in the framework.
    """

    def __init__(self):
        """
        Initialize the SoftMax layer. This layer does not have trainable parameters.
        """
        super().__init__()

    def forward(self, input_tensor):
        """
        Perform the forward pass of the SoftMax activation.

        Args:
            input_tensor (np.ndarray): The input tensor (logits) to transform.

        Returns:
            np.ndarray: The probability distribution for each element of the batch.
        """
        # Apply the SoftMax function in a numerically stable way
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output_tensor = probabilities
        return probabilities

    def backward(self, error_tensor):
        """
        Perform the backward pass of the SoftMax activation.

        Args:
            error_tensor (np.ndarray): The error tensor from the next layer.

        Returns:
            np.ndarray: The error tensor for the previous layer, computed using the Jacobian matrix.
        """
        # Compute the gradient for each sample in the batch
        batch_size = self.output_tensor.shape[0]
        gradient = np.zeros_like(error_tensor)

        for i in range(batch_size):
            # Diagonal matrix of the softmax outputs
            softmax_diag = np.diag(self.output_tensor[i])
            # Outer product of the softmax outputs
            softmax_outer = np.outer(self.output_tensor[i], self.output_tensor[i])
            # Jacobian matrix of the softmax function
            jacobian = softmax_diag - softmax_outer
            gradient[i] = np.dot(jacobian, error_tensor[i])

        return gradient
