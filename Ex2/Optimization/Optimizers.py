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
    """
    Stochastic Gradient Descent (SGD) with Momentum optimizer.

    This optimizer updates the weights by considering both the current gradient
    and a fraction of the previous update. The momentum term helps accelerate
    updates in the direction of decreasing loss, smoothing out oscillations.

    Attributes:
        learning_rate (float): The learning rate for the optimization step.
        momentum_rate (float): The momentum factor controlling the influence of previous gradients.
        velocity (np.ndarray or None): The velocity term that accumulates the past updates.
    """

    def __init__(self, learning_rate, momentum_rate):
        """
        Initialize the SGD with Momentum optimizer.

        Args:
            learning_rate (float): The step size for each weight update.
            momentum_rate (float): The momentum factor (commonly between 0 and 1).
        """
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculate the updated weights using SGD with Momentum.

        Args:
            weight_tensor (np.ndarray): The current weights of the model.
            gradient_tensor (np.ndarray): The gradient of the loss with respect to the weights.

        Returns:
            np.ndarray: The updated weights after applying the SGD with Momentum step.
        """
        if self.velocity is None:
            # Initialize velocity as zeros with the same shape as weight_tensor
            self.velocity = np.zeros_like(weight_tensor)

        # Update velocity and then use it to update weights
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        return weight_tensor + self.velocity

class Adam:
    """
    Adaptive Moment Estimation (Adam) optimizer.

    Adam adapts the learning rate for each parameter by using estimates of 
    first and second moments of the gradients. It usually requires little 
    tuning and often works well in practice.

    Attributes:
        learning_rate (float): The step size for each weight update.
        mu (float): The exponential decay rate for the first moment estimates.
        rho (float): The exponential decay rate for the second moment estimates.
        epsilon (float): A small constant for numerical stability to avoid division by zero.
        v (np.ndarray or None): The first moment vector (moving average of gradients).
        r (np.ndarray or None): The second moment vector (moving average of squared gradients).
        k (int): The time step counter, incremented after each update.
    """

    def __init__(self, learning_rate, mu=0.9, rho=0.999):
        """
        Initialize the Adam optimizer.

        Args:
            learning_rate (float): The step size for each weight update.
            mu (float, optional): The first moment decay rate. Defaults to 0.9.
            rho (float, optional): The second moment decay rate. Defaults to 0.999.
        """
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = 1e-8
        self.v = None
        self.r = None
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculate the updated weights using the Adam optimization algorithm.

        Args:
            weight_tensor (np.ndarray): The current weights of the model.
            gradient_tensor (np.ndarray): The gradient of the loss with respect to the weights.

        Returns:
            np.ndarray: The updated weights after applying the Adam optimization step.
        """
        # Initialize first and second moment vectors if they are not set yet
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)
        # Increment the time step
        self.k += 1

        # Update biased first moment estimate
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor

        # Update biased second moment estimate
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.v / (1 - self.mu ** self.k)

        # Compute bias-corrected second moment estimate
        v_hat = self.r / (1 - self.rho ** self.k)

        # Update the weights
        return weight_tensor - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
