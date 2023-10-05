from numpy import ndarray
from operation import Operation
import numpy as np

class Sigmoid(Operation):
    '''
    Função de ativação de sigmoide
    '''
    def __init__(self):
        super().__init__()

    def _evaluate_output(self) -> ndarray:
        return 1 / (1 + np.exp(-1 * self.input_))
    
    def _evaluate_input_grad(self, output_grad: ndarray) -> ndarray:
        sigmoid_backward = self.output * (1 - self.output)
        input_grad = sigmoid_backward * output_grad
        
        return input_grad