import copy

class NeuralNetwork:
    """
    NeuralNetwork defines the entire architecture and manages the training and testing process.
    
    Attributes:
        optimizer: The optimizer object used for updating weights.
        loss (list): Stores the loss value for each iteration during training.
        layers (list): Holds the network's architecture (all layers).
        data_layer: Provides input data and labels.
        loss_layer: Computes loss and predictions.
        label_tensor: Stores the label tensor for use in backpropagation.
        weights_initializer: Initializer for weights.
        bias_initializer: Initializer for biases.
    """

    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None):
        """
        Initialize the NeuralNetwork with an optimizer and empty lists for layers and loss values.
        
        Args:
            optimizer: The optimizer object for the network.
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
        Perform a forward pass through the network using input from the data layer.
        
        Returns:
            The output of the last layer (loss layer) of the network.
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
        Propagate the input tensor through the network and return the prediction.
        
        Args:
            input_tensor: The input tensor to test the network.
        
        Returns:
            The prediction of the last layer.
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  # Pass through each layer
        return input_tensor
