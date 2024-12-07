import numpy as np

class Constant:
    """
    A weight initializer that sets all values to a constant.
    
    Attributes:
        value (float): The constant value to fill the weights with.
    """

    def __init__(self, value=0.1):
        """
        Initialize the Constant initializer.

        Args:
            value (float, optional): The constant value to initialize weights with. Defaults to 0.1.
        """
        self.value = value

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        """
        Initialize the weights to a constant value.

        Args:
            weights_shape (tuple): The shape of the weights to be initialized.
            fan_in (int, optional): The number of input units (not used in this initializer).
            fan_out (int, optional): The number of output units (not used in this initializer).

        Returns:
            np.ndarray: A numpy array of shape `weights_shape` filled with the constant value.
        """
        return np.full(weights_shape, self.value)

class UniformRandom:
    """
    A weight initializer that sets all values from a uniform distribution in [0, 1].
    """

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        """
        Initialize the weights uniformly at random in [0, 1].

        Args:
            weights_shape (tuple): The shape of the weights to be initialized.
            fan_in (int, optional): The number of input units (not used in this initializer).
            fan_out (int, optional): The number of output units (not used in this initializer).

        Returns:
            np.ndarray: A numpy array of shape `weights_shape` filled with values sampled uniformly from [0, 1].
        """
        return np.random.uniform(0, 1, weights_shape)

class Xavier:
    """
    A weight initializer that implements Xavier initialization.
    
    The weights are drawn from a normal distribution with mean 0 and 
    a standard deviation of sqrt(2 / (fan_in + fan_out)).
    """

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize the weights using Xavier initialization.

        Args:
            weights_shape (tuple): The shape of the weights to be initialized.
            fan_in (int): The number of input units.
            fan_out (int): The number of output units.

        Returns:
            np.ndarray: A numpy array of shape `weights_shape` initialized with the Xavier scheme.
        """
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, std, weights_shape)

class He:
    """
    A weight initializer that implements He initialization.
    
    The weights are drawn from a normal distribution with mean 0 and 
    a standard deviation of sqrt(2 / fan_in).
    """

    def initialize(self, weights_shape, fan_in, fan_out=None):
        """
        Initialize the weights using He initialization.

        Args:
            weights_shape (tuple): The shape of the weights to be initialized.
            fan_in (int): The number of input units.
            fan_out (int, optional): The number of output units (not used in this initializer).

        Returns:
            np.ndarray: A numpy array of shape `weights_shape` initialized with the He scheme.
        """
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, weights_shape)
