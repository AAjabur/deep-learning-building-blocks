from operations import ParamOperation
from numpy import ndarray
import numpy as np

class WeightMultiply(ParamOperation):
    '''
    Operação que multiplica a entrada por pesos
    '''

    def __init__(self, W: ndarray):
        super().__init__()

    def _evaluate_output(self) -> ndarray:
        return self.input_ @ self.param
    
    def _evaluate_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad @ self.param.T
    
    def _evaluate_param_grad(self, output_grad: ndarray) -> ndarray:
        return self.input_.T @ output_grad

