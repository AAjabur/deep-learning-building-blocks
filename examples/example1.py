import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
lib_dir = os.path.join(parent_dir, "lib")
sys.path.append(lib_dir)

from layers import DenseLayer
from losses import MeanSquaredError
from operations import Identity
from neural_network import NeuralNetwork

my_net = NeuralNetwork([DenseLayer(1, Identity())], MeanSquaredError())