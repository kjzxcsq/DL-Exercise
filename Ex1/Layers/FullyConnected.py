import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        self.input_tensor = np.hstack([input_tensor, np.ones((batch_size, 1))])
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        gradient_input = np.dot(error_tensor, self.weights[:-1, :].T)
        
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        
        return gradient_input

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self._gradient_weights
