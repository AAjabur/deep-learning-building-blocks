from param_operation import Operation
from numpy import ndarray

class Identity(Operation):
    '''
    Operador que não faz nada
    '''

    def __init__(self):
        super().__init__()

    def _evaluate_output(self) -> ndarray:
        return self.input_
    
    def _evaluate_input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad
