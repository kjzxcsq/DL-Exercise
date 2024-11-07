import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output_tensor = probabilities
        return probabilities

    def backward(self, error_tensor):
        # Compute the gradient for the softmax function
        batch_size = self.output_tensor.shape[0]
        gradient = np.zeros_like(error_tensor)

        # Iterate over each sample in the batch
        for i in range(batch_size):
            # Create a diagonal matrix of the softmax outputs
            softmax_diag = np.diag(self.output_tensor[i])
            softmax_outer = np.outer(self.output_tensor[i], self.output_tensor[i])
            jacobian = softmax_diag - softmax_outer
            gradient[i] = np.dot(jacobian, error_tensor[i])

        return gradient
