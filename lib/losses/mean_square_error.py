from losses import Loss
from numpy import ndarray
import numpy as np

class MeanSquaredError(Loss):
    def __init__(self) -> None:
        super().__init__()

    def _evaluate_output(self) -> float:
        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]
        return loss
    
    def _evaluate_input_grad(self) -> ndarray:
        return 2 * (self.prediction - self.target) / self.prediction.shape[0]