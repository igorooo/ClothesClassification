import numpy as np
from Layer import Layer_


class FF_layer(Layer_):

    gauss_MEAN = 0
    gauss_ST_DEVIATION = 1

    def __init__(self,input_size, nodes, dropout_prob, isLast = False):
        super(FF_layer, self).__init__()
        self.i_input_size = input_size
        self.t_weights_dim = (input_size,nodes)
        self.i_input_size = input_size
        self.i_num_nodes = nodes
        self.i_dropout_prob = dropout_prob
        self.b_isLast = isLast

        self.mx_weights = None
        self.v_bias = None

        if isLast:
            self.layer_type = Layer_.last_FF_layer
        if not isLast:
            self.layer_type = Layer_.FF_layer







    def __init_random_weights__(self):
        self.mx_weights = np.random.normal(FF_layer.gauss_MEAN, FF_layer.gauss_ST_DEVIATION, self.t_weights_dim)
        self.v_bias = np.random.normal(FF_layer.gauss_MEAN, FF_layer.gauss_ST_DEVIATION, self.i_num_nodes)

    def forwardPass(self, image):
        #if self.v_weights == None or self.v_bias == None:
         #   raise Exception('FF_layer Exception: Weights not initialized!')

        self.input = image

        res = (image@self.mx_weights) + self.v_bias
        self.values_1 = res
        res = self.__relu__(res)
        self.values_2 = res

        if self.b_isLast:
            res = self.__softmax__(res)
        else:
            res = self.__dropout__(res,self.i_dropout_prob)


        self.result = res
        #print(res.shape)
        return res

    def backwardPass(self,dL, alfa):
        w = self.mx_weights
        prev_l, cur_l = w.shape

        y = self.input.reshape(self.input.shape[0],1)
        dFi = dL * self.v_relu_grad.reshape(dL.shape)

        dropout = self.v_dropout_grad.reshape(dFi.shape)
        dFi = dFi * dropout

        dY = np.dot(w,dFi)
        dB = dFi

        dW = np.dot(y, dFi.T)

        return dW, dB, dY









        """

        dFi = np.multiply(self.v_relu_grad, dL[:,0])
        dFi = np.multiply(self.v_dropout_grad, dFi)
        jW = np.ones(self.mx_weights.shape)


        print(self.input.shape,end=' input')
        print(jW.shape)
        dw = self.elWiseMult(jW,self.input.reshape(self.input.shape[0], 1))
        dW = self.elWiseMult(dw, dFi)
        dB = dFi
        dY = np.sum(self.mx_weights,axis=1)
        print(dY.shape,end='dy FF')
        dY = self.elWiseMult(dY, dFi.reshape(dFi.shape[0], 1))

        self.v_bias += (alfa*dB)
        self.mx_weights += (alfa*dW)

        return dY
        """


    def __dropout__(self,X, p):
        binomial = np.random.binomial(1,1.-p,X.shape[0])
        self.v_dropout_grad = binomial
        return np.multiply(X,binomial)

    def __softmax__(self, w):
        res = np.exp(w)
        divider = np.sum(res)
        if divider == 0:
            divider = 1
        return res/divider
