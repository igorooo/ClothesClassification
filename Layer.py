import numpy as np
import abc




class Layer_():

    #Layer types:
    last_FF_layer = 3
    FF_layer = 2
    CNN_layer = 1

    def __init__(self):
        self.result = None
        self.t_input_size = (0)
        self.input = None
        self.layer_type = ''
        self.values_1 = None
        self.values_2 = None
        self.values_3 = None
        self.v_relu_grad = None
        self.v_dropout_grad = None

    def forwardPass(self, image):
        pass

    def backwardPass(self,dL, alfa):
        pass

    def __relu__(self,x):
        z = np.zeros_like(x)
        self.v_relu_grad = np.where(x > z, 1, 0)
        return np.where(x > z, x, z)

    def checkResultSize(self):
        return np.shape(self.forwardPass(np.zeros(self.t_input_size)))



