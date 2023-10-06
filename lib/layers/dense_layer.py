from layers import Layer
from numpy import ndarray
import numpy as np
from operations import Sigmoid, WeightMultiply, BiasAdd, Operation

class DenseLayer(Layer):
    def __init__(self, num_of_neurons: int, activation: Operation) -> None:
        super().__init__(num_of_neurons, activation)

    def _setup_layer(self, input: ndarray):
        self.params = [
            np.random.randn(input.shape[1], self.num_of_neurons), # Pesos
            np.random.randn(1, self.num_of_neurons)               # Bias
        ]

        self.operations = [
            WeightMultiply(self.params[0]), 
            BiasAdd(self.params[1]),
            self.activation
        ]


    