from operations.operation import Operation
from operations.param_operation import ParamOperation
from numpy import ndarray
import numpy as np
from typing import List

class Layer:
    '''
    Um layer em uma rede neural. Um layer é definido pelas operações que ocorrem nela,
    na ordem em que elas ocorrem
    '''

    def __init__(self, num_of_neurons: int, activation: Operation) -> None:
        self.num_of_neurons = num_of_neurons # número de neurônios de saída
        self.activation = activation # função de ativação dessa layer
        self.first = True
        self.params: List[ndarray] = [] # lista que armazenará os parâmetros dos operadores que forem
                                        # parametrizáveis na ordem em que os operadores aparecem
        self.param_grads: List[ndarray] = [] # lista que armazenará a derivada parcial do erro em relação
                                             # aos parâmetros dos operadores parametrizáveis na ordem em que
                                             # os operadores aparecem
        self.operations: List[Operation] = []

    def _setup_layer(self, **kwargs):
        '''
        Função que faz o setup desse layer, define os operadores que o layer utiliza
        e define os parâmetros dos operadores que precisam de parâmetros
        '''
        raise NotImplementedError()
    
    def forward_pass(self, input: ndarray) -> ndarray:
        if self.first:
            self._setup_layer(input)

        self.input_ = input

        for operation in self.operations:
            input = operation.forward_pass(input)

        self.output = input

        return self.output
    
    def backward_pass(self, output_grad: ndarray) -> ndarray:
        assert self.output.shape == output_grad.shape

        for operation in reversed(self.operations):
            output_grad = operation.backward_pass(output_grad)

        input_grad = output_grad

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)
