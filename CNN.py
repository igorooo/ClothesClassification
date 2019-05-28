import numpy as np
import ConvLayer as cl
import FullyConnectedLayer as fl
import Layer



class CNN():

    def __init__(self,image_size = 36, num_conv_layers = 2, num_filters = (8,8), filters_size = (3,3), pooling_dim = (2,2), padding = (True, True), num_fflayers = 2, nodes_fflayers = (8,8), dropout_prob = (0.2,0)):

        if not (num_conv_layers == np.shape(num_filters)[0] == np.shape(filters_size)[0] == np.shape(pooling_dim)[0]):
            raise Exception('CNN Exception: Parameters for convolutional layers are not consistent')
        if not (num_fflayers == np.shape(nodes_fflayers)[0] == np.shape(dropout_prob)[0]):
            raise Exception('CNN Exception: Parameters for fully connected layers are not consistent')


        self.i_image_size = image_size
        self.i_num_conv_layers = num_conv_layers
        self.t_num_filters = num_filters
        self.t_filter_size = filters_size
        self.t_pooling_dim = pooling_dim
        self.t_padding = padding
        self.i_num_fflayers = num_fflayers
        self.t_nodes_fflayers = nodes_fflayers
        self.t_dropout_prob = dropout_prob

        self.l_layers = []

        self.t_last_input_size = None





    def __init_cnn_layers__(self):

        image_size = (self.i_image_size, self.i_image_size, 1)
        cnnLayer = None

        for i in range( self.i_num_conv_layers ):
            filter_size = (self.t_filter_size[i], self.t_filter_size[i], image_size[-1])
            cnnLayer = cl.Conv_layer(image_size, filter_size, self.t_num_filters[i], self.t_pooling_dim[i], self.t_padding[i])
            cnnLayer.__init_random_filters__()

            self.l_layers.append(cnnLayer)
            image_size = cnnLayer.checkResultSize()
        cnnLayer.b_isLast = True

        self.t_last_input_size = image_size

    def __init_ff_layers__(self):
        image_size = np.prod(self.t_last_input_size)

        for i in range( self.i_num_fflayers ):

            fnnlayer = fl.FF_layer(image_size,self.t_nodes_fflayers[i],self.t_dropout_prob[i])
            fnnlayer.__init_random_weights__()
            image_size = self.t_nodes_fflayers[i]
            self.l_layers.append(fnnlayer)



    def forwardPass(self, image):

        #i = 0

        for layer in self.l_layers:
            image = layer.forwardPass(image)

        return self.__classify__(image)









    def __classify__(self, x):
        return np.argmax(x,axis=-1)









