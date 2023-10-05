from param_operation import ParamOperation
from numpy import ndarray
import numpy as np

class BiasAdd(ParamOperation):
    '''
    Operador que soma um bias na nossa entrada
    '''

    def __init__(self, bias: ndarray) -> ndarray:
        assert bias.shape[0] == 1 # garante que o mesmo Bias será aplicado à todas as observações

        super().__init__(bias)

    def _evaluate_output(self) -> ndarray:
        return self.input_ + self.param
    
    def _evaluate_input_grad(self, output_grad: ndarray) -> ndarray:
        return np.ones_like(self.input_) * output_grad
    
    def _evaluate_param_grad(self, output_grad: ndarray) -> ndarray:
        param_grad = np.ones_like(self.param) * output_grad

        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
