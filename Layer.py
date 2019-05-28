import numpy as np
import abc

class Layer_():

    def __init__(self):
        self.result = None
        self.t_input_size = (0)

    def forwardPass(self, image):
        pass

    def __relu__(self,x):
        z = np.zeros_like(x)
        return np.where(x > z, x, z)

    def checkResultSize(self):
        return np.shape(self.forwardPass(np.zeros(self.t_input_size)))
