import numpy as np
import ConvLayer as cl
import FullyConnectedLayer as fl
import Layer



class CNN():

    def __init__(self,image_size = 36, num_conv_layers = 2, num_filters = (8,8), filters_size = (3,3), pooling_dim = (2,2), padding = (False,False), num_fflayers = 2, nodes_fflayers = (30,10), dropout_prob = (0.2,0)):

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





    def init_random(self):
        self.__init_cnn_layers__()
        self.__init_ff_layers__()



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

    def __forwardPass__(self, image):

        i = 0
        img = image


        for layer in self.l_layers:
            i += 1
            img = layer.forwardPass(img)


        return self.__classify_vector__(img)

    def __backwardPass__(self, dL, learningRate):

        lay = self.l_layers
        dl = dL

        for layer in reversed(lay):
            res = layer.backwardPass(dl)
            layer.update(res[0], res[1], learningRate)
            dl = res[2]

        return




    def learn(self, training_set, epochs = 1000, learning_rate = 0.001):

        x_tr_set = training_set[0].shape[0]


        for ep in range(epochs):

            ix = np.random.randint(0,x_tr_set)

            img = training_set[0][ix,:,:]
            label = training_set[1][ix]


            result = self.__forwardPass__(img)
            dL = self.__cross_entropy__(label, result)

            self.__backwardPass__(dL,learning_rate)

    def vaidation(self,valid_set):
        x_valid_set = valid_set[0].shape[0]

        positive = 0

        for i in range(50):

            result = self.forwardPass(valid_set[0][i,:,:])

            if(result == np.argmax(valid_set[1][i])):
                positive += 1

        return positive/x_valid_set




    @staticmethod
    def __classify__(x):
        return np.argmax(x,axis=-1)

    @staticmethod
    def __classify_vector__(x):
        res = np.zeros((10))
        res[np.argmax(x,axis=-1)] = 1
        return res

    @staticmethod
    def __cross_entropy__(label, y):
        label = label.reshape(y.shape)
        div = y - label
        return np.reshape(label * div,(y.shape[0], 1))



