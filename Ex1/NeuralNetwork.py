import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer  # Optimizer object for the network
        self.loss = []  # List to store the loss values for each iteration
        self.layers = []  # List to hold the network's architecture
        self.data_layer = None  # Data layer to provide input data and labels
        self.loss_layer = None  # Loss layer to compute loss and prediction
        self.label_tensor = None  # Variable to store label tensor

    def forward(self):
        """
        Perform a forward pass through the network using input from the data layer.
        :return: The output of the last layer (loss layer) of the network
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
        :param layer: The layer to be appended to the network
        """
        if hasattr(layer, 'trainable') and layer.trainable:  # Check if the layer is trainable
            layer.optimizer = copy.deepcopy(self.optimizer)  # Make a deep copy of the optimizer
        self.layers.append(layer)  # Append the layer to the layers list

    def train(self, iterations):
        """
        Train the network for a specified number of iterations.
        :param iterations: Number of iterations to train the network
        """
        for _ in range(iterations):
            output_tensor = self.forward()  # Perform forward pass
            loss_value = self.loss_layer.forward(output_tensor, self.label_tensor)  # Compute loss using stored labels
            self.loss.append(loss_value)  # Store loss value
            self.backward()  # Perform backward pass

    def test(self, input_tensor):
        """
        Propagate the input tensor through the network and return the prediction.
        :param input_tensor: The input tensor to test the network
        :return: The prediction of the last layer
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)  # Pass through each layer
        return input_tensor

