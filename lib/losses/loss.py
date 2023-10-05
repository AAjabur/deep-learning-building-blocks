from numpy import ndarray
import numpy as np

class Loss:
    '''
    Classe abstrata para a definição do erro em uma rede neural
    '''
    def __init__(self) -> None:
        pass

    def forward_pass(self, prediction: ndarray, target: ndarray) -> float:
        '''
        Calcula o erro total entre as previsões e os alvos
        '''
        assert prediction.shape == target.shape

        self.prediction = prediction
        self.target = target

        return self._evaluate_output()
    
    def backward_pass(self) -> ndarray:
        '''
        Calcula a derivada parcial do erro em relação às previsões
        '''
        self.input_grad = self._evaluate_input_grad()

        assert self.prediction.shape == self.input_grad.shape

        return self.input_grad


    def _evaluate_output(self) -> float:
        '''
        Calcula o erro a partir das previsões e dos valores alvo, esses valores
        podem ser acessados com os atributos self.prediction e self.target

        ** OBRIGATÓRIO IMPLEMENTAR **
        '''
        raise NotImplementedError()
    
    def _evaluate_input_grad(self) -> ndarray:
        '''
        Calcula a derivada parcia do erro em relação aos valores alvo, esses valores
        podem ser acessados com os atributos self.prediction e self.target

        ** OBRIGATÓRIO IMPLEMENTAR **
        '''
        raise NotImplementedError()
