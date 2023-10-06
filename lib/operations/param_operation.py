from operations import Operation
from numpy import ndarray

class ParamOperation(Operation):
    '''
    Uma operação que recebe parâmetros, por exemplo uma multiplicação de matrizes receberá uma matriz
    de entrada e uma matriz W que será multiplicada por ela, essa matriz W é um parâmetro
    '''
    def __init__(self, param: ndarray) -> ndarray:
        super().__init__()
        self.param = param

    def backward_pass(self, output_grad: ndarray) -> ndarray:
        '''
        Responsável por calcular a derivada parcial do erro em relação ao input dessa operação
        e a derivada parcial do erro em relação ao parâmetro dessa operação

        output_grad: deriavada parcial do erro em relação ao output dessa operação
        '''

        assert self.output.shape == output_grad.shape

        self.input_grad = self._evaluate_input_grad(output_grad)
        self.param_grad = self._evaluate_param_grad(output_grad)

        assert self.input_.shape == self.input_grad.shape
        assert self.param.shape == self.param_grad.shape

    def _evaluate_param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Função que calcula a derivada parcial do erro em relação ao parâmetro do operador, deve
        ser definida pelo operador
        '''
        raise NotImplementedError()