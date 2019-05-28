import numpy as np
from Layer import Layer_


class FF_layer(Layer_):

    gauss_MEAN = 0
    gauss_ST_DEVIATION = 1

    def __init__(self,input_size, nodes, dropout_prob):
        super(FF_layer, self).__init__()
        self.i_input_size = input_size
        self.t_weights_dim = (input_size,nodes)
        self.i_input_size = input_size
        self.i_num_nodes = nodes
        self.i_dropout_prob = dropout_prob

        self.mx_weights = None
        self.v_bias = None




    def __init_random_weights__(self):
        self.mx_weights = np.random.normal(FF_layer.gauss_MEAN, FF_layer.gauss_ST_DEVIATION, self.t_weights_dim)
        self.v_bias = np.random.normal(FF_layer.gauss_MEAN, FF_layer.gauss_ST_DEVIATION, self.i_num_nodes)

    def forwardPass(self, image):
        #if self.v_weights == None or self.v_bias == None:
         #   raise Exception('FF_layer Exception: Weights not initialized!')

        res = (image@self.mx_weights) + self.v_bias
        res = self.__relu__(res)
        res = self.__dropout__(res,self.i_dropout_prob)
        res = self.__softmax__(res)
        self.result = res
        return res

    def __dropout__(self,X, p):
        binomial = np.random.binomial(1,1.-p,X.shape[0])
        return np.multiply(X,binomial)

    def __softmax__(self, w):
        res = np.exp(w)
        divider = np.sum(res)
        return res/divider









