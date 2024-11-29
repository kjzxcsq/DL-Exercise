import numpy as np

class Sgd:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Attributes:
        learning_rate (float): The learning rate for the gradient descent updates.
    """

    def __init__(self, learning_rate: float):
        """
        Initialize the SGD optimizer with a learning rate.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculate the updated weights using the basic gradient descent update rule.

        Args:
            weight_tensor: The current weights tensor.
            gradient_tensor: The gradient tensor to apply for the update.

        Returns:
            Updated weight tensor after applying gradient descent.
        """
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.velocity

class Adam:
    def __init__(self, learning_rate, mu=0.9, rho=0.999):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.v = None
        self.r = None
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)
        self.k += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2)
        m_hat = self.v / (1 - self.mu ** self.k)
        v_hat = self.r / (1 - self.rho ** self.k)
        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
