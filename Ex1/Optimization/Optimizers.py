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