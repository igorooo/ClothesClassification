import numpy as np
import ConvLayer as cl
import Layer



class CNN():

    def __init__(self,image_size = 36, num_conv_layers = 2, num_filters = (8,8), filters_size = (3,3), pooling_dim = (2,2), padding = (True, True), num_fflayers = 2, nodes_fflayers = (8,8), dropout_prob = (0.2,0.2)):

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





    def __init_cnn_layers__(self):

        for i in range(self.i_num_conv_layers):
            cnnLayer = cl.Conv_layer(self.i_image_size, self.t_filter_size[i], self.t_num_filters[i], self.t_pooling_dim[i], self.t_padding[i])
            self.l_layers.append(cnnLayer)

    def __init_ff_layers__(self):









