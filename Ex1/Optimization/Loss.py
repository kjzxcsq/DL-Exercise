import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        """
        Computes the CrossEntropy Loss over the batch without dividing by batch_size.

        :param prediction_tensor: Output from the SoftMax layer, shape: (batch_size, num_classes)
        :param label_tensor: Ground truth labels in one-hot encoding, shape: (batch_size, num_classes)
        :return: The scalar loss value accumulated over the batch
        """
        # Use epsilon for numerical stability
        epsilon = np.finfo(float).eps
        self.prediction_tensor = prediction_tensor + epsilon  # Add epsilon to avoid log(0)
        self.label_tensor = label_tensor

        # Compute cross-entropy loss without dividing by batch_size
        loss = -np.sum(label_tensor * np.log(self.prediction_tensor))
        return loss

    def backward(self, label_tensor):
        """
        Computes the error tensor for backpropagation without dividing by batch_size.

        :return: The error tensor for the previous layer, shape: (batch_size, num_classes)
        """
        # Use epsilon for numerical stability
        epsilon = np.finfo(float).eps
        error_tensor = - (label_tensor / (self.prediction_tensor + epsilon))
        return error_tensor
