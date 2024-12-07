import copy

class NeuralNetwork:
    """
    A neural network class that defines the network architecture and manages
    training and testing procedures.

    This class holds a sequence of layers, an optimizer, and initializer objects.
    It coordinates forward and backward passes, applying the optimizer for
    trainable parameters in each layer, and maintains a log of loss values
    during training.

    Attributes:
        optimizer: The optimizer object for updating the network weights.
        loss (list): A list to store the loss value for each training iteration.
        layers (list): A list holding all layers of the network.
        data_layer: A data provider that returns input data and labels.
        loss_layer: A layer that computes the loss and potentially the final
            predictions of the network.
        label_tensor: A tensor containing the target labels for training.
        weights_initializer: An initializer object for layer weights.
        bias_initializer: An initializer object for layer biases.
    """

    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None):
        """
        Initialize the NeuralNetwork instance with a given optimizer and optional
        weight and bias initializers.

        Args:
            optimizer: The optimizer to be used for updating layer parameters.
            weights_initializer (optional): An initializer object for weights.
            bias_initializer (optional): An initializer object for biases.
        """
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        """
        Perform a forward pass through the entire network.

        This method retrieves the next batch of input data and labels from the
        data_layer, then propagates the input forward through all layers. 
        Finally, the output is returned.

        Returns:
            np.ndarray: The network output after the forward pass.
        """
        input_tensor, self.label_tensor = self.data_layer.next()  # Get input and labels from data layer
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  # Pass through each layer
        return input_tensor

    def backward(self):
        """
        Perform a backward pass starting from the loss layer and propagate backward through the network.
        """
        error_tensor = self.loss_layer.backward(self.label_tensor)  # Compute initial error tensor using stored labels
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)  # Propagate backward through each layer

    def append_layer(self, layer):
        """
        Append a layer to the network and set the optimizer for trainable layers.
        
        Args:
            layer: The layer to be appended to the network.
        """
        if hasattr(layer, 'trainable') and layer.trainable:  # Check if the layer is trainable
            layer.optimizer = copy.deepcopy(self.optimizer)  # Make a deep copy of the optimizer
            if self.weights_initializer and self.bias_initializer:
                layer.initialize(self.weights_initializer, self.bias_initializer)  # Initialize weights and biases
        self.layers.append(layer)  # Append the layer to the layers list

    def train(self, iterations):
        """
        Train the network for a specified number of iterations.
        
        Args:
            iterations: Number of iterations to train the network.
        """
        for _ in range(iterations):
            output_tensor = self.forward()  # Perform forward pass
            loss_value = self.loss_layer.forward(output_tensor, self.label_tensor)  # Compute loss using stored labels
            self.loss.append(loss_value)  # Store loss value
            self.backward()  # Perform backward pass

    def test(self, input_tensor):
        """
        Test the network with a given input tensor.

        Args:
            input_tensor (np.ndarray): The input data for testing.

        Returns:
            np.ndarray: The network prediction for the given input.
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  # Pass through each layer
        return input_tensor
