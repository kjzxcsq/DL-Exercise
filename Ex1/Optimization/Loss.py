import numpy as np

class CrossEntropyLoss:
    """
    CrossEntropyLoss computes the cross-entropy loss for classification tasks, 
    typically used in conjunction with the SoftMax activation function.
    """

    def __init__(self):
        """
        Initialize the CrossEntropyLoss class.
        """
        self.prediction_tensor = None
        self.label_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        """
        Compute the CrossEntropy loss over the batch without dividing by the batch size.

        Args:
            prediction_tensor (np.ndarray): Output from the SoftMax layer, shape: (batch_size, num_classes).
            label_tensor (np.ndarray): Ground truth labels in one-hot encoding, shape: (batch_size, num_classes).

        Returns:
            float: The scalar loss value accumulated over the batch.
        """
        # Use epsilon for numerical stability to prevent log(0)
        epsilon = np.finfo(float).eps
        self.prediction_tensor = prediction_tensor + epsilon
        self.label_tensor = label_tensor

        # Compute the cross-entropy loss
        loss = -np.sum(label_tensor * np.log(self.prediction_tensor))
        return loss

    def backward(self, label_tensor):
        """
        Compute the error tensor for backpropagation without dividing by the batch size.

        Args:
            label_tensor (np.ndarray): Ground truth labels in one-hot encoding.

        Returns:
            np.ndarray: The error tensor for the previous layer, shape: (batch_size, num_classes).
        """
        # Use epsilon for numerical stability to avoid division by zero
        epsilon = np.finfo(float).eps
        error_tensor = - (label_tensor / (self.prediction_tensor + epsilon))
        return error_tensor