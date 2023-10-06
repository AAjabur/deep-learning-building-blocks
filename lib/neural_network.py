from typing import List
from losses import Loss
from layers import Layer
from numpy import ndarray
import numpy as np


class NeuralNetwork:
    def __init__(self, layers: List[Layer], loss: Loss) -> None:
        self.layers = layers
        self.loss = loss

    def forward_pass(self, x_batch: ndarray) -> ndarray:
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward_pass(x_out)

        return x_out
    
    def backward_pass(self, loss_grad: ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward_pass(grad)

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        predictions = self.forward_pass(x_batch)
        loss = self.loss.forward_pass(predictions, y_batch)

        self.backward_pass(self.loss.backward_pass())

        return loss

    def get_params(self):
        for layer in self.layers:
            yield from layer.params

    def get_param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads
