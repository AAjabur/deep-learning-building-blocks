from numpy import ndarray
import numpy as np

class Operation(object):
    '''
    Classe base para uma operação em uma rede neural.
    '''
    def __init__(self):
        pass

    def forward_pass(self, input: ndarray) -> ndarray:
        '''
        Responsável por calcular o valor de saída da função e armazena-lo
        '''
        self.input_ = input
        self.output = self._evaluate_output()

        return self.output
    
    def backward_pass(self, output_grad: ndarray) -> ndarray:
        '''
        Responsável por calcular a derivada parcial do erro em relação ao input dessa operação

        output_grad: deriavada parcial do erro em relação ao output dessa operação
        '''
        assert self.output.shape == output_grad.shape

        self.input_grad = self._evaluate_input_grad(output_grad)

        assert self.input_.shape == self.input_grad.shape

    def _evaluate_output(self) -> ndarray:
        '''
        Função que calcula a saída da operação, deve ser definida pelo operador,
        a entrada da função pode ser acessada por self.input_
        '''
        raise NotImplementedError()
    
    def _evaluate_input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Função que calcula a derivada parcial do erro em relação ao input do operador, deve
        ser definida pelo operador
        '''
        raise NotImplementedError()
