class BaseLayer:
    """
    Base class for all layers in the deep learning framework.

    Attributes:
        trainable (bool): Indicates whether the layer has trainable parameters.
    """

    def __init__(self):
        """
        Initialize the BaseLayer with the trainable attribute set to False.
        """
        self.trainable = False