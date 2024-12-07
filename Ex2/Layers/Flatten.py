from .Base import BaseLayer

class Flatten(BaseLayer):
    """
    A Flatten layer that reshapes the input tensor into a 2D matrix (batch_size x features).

    This layer takes an input of shape (batch_size, ..., features) and flattens it into
    (batch_size, -1), where -1 represents all other dimensions combined.

    Attributes:
        input_shape (tuple): The shape of the input tensor before flattening.
    """

    def __init__(self):
        """
        Initialize the Flatten layer.
        """
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        """
        Perform the forward pass of the Flatten layer.

        This method flattens the input tensor while preserving the batch size dimension.

        Args:
            input_tensor (np.ndarray): The input tensor of shape (batch_size, ...).

        Returns:
            np.ndarray: The flattened output tensor of shape (batch_size, -1).
        """
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(self.input_shape[0], -1)

    def backward(self, error_tensor):
        """
        Perform the backward pass of the Flatten layer.

        This method reshapes the error tensor back to the original input shape, 
        so it can be properly passed to the previous layer in the network.

        Args:
            error_tensor (np.ndarray): The error tensor from the subsequent layer
                                       of shape (batch_size, -1).

        Returns:
            np.ndarray: The reshaped error tensor of the original input shape.
        """
        return error_tensor.reshape(self.input_shape)
