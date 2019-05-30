import numpy as np
from Layer import Layer_

class Conv_layer(Layer_):

    gauss_MEAN = 0
    gauss_ST_DEVIATION = 1


    def __init__(self, input_img_size, filter_size, num_of_filters, pooling_dim = 2, padding = False, isLast = False):
        super(Conv_layer, self).__init__()
        self.t_input_img_size = np.array(input_img_size)  #delete
        self.t_input_size = input_img_size
        self.t_filter_size = np.array(filter_size)
        self.i_num_of_filters = num_of_filters
        self.i_pooling_dim = pooling_dim
        self.b_padding = padding
        self.b_isLast = isLast

        self.layer_type = Layer_.CNN_layer

        if self.b_padding:
            self.t_conv_size = (self.t_input_img_size[0] + self.t_filter_size[0] -1,self.t_input_img_size[0] + self.t_filter_size[0] -1)

        if not self.b_padding:
            self.t_conv_size = (self.t_input_img_size[0] - self.t_filter_size[0] +1,self.t_input_img_size[0] - self.t_filter_size[0] +1)

        self.t_Allfilters_size = (self.t_filter_size[0], self.t_filter_size[1], self.t_filter_size[2], self.i_num_of_filters)


        self.mx_filters = None
        self.v_bias = None

        self.mx_max_pool_map = None
        self.features = []





    def forwardPass(self, image):
        if len(self.mx_filters) == 0 or len(self.v_bias) == 0:
            raise Exception('Conv_layer Exception: Weights non initialized!')




        image = self.__move_to_3D__(image)
        self.input = image

        features = self.__cnn_layer__(image)
        self.values_1 = features
        features = self.__relu__(features)
        self.values_2 = features
        features = self.__max__pooling__(features)
        self.values_3 = features

        if(self.b_isLast):
            features = features.flatten()

        self.result = features

        return features

    def backwardPass(self,dL):

        w = np.array(self.mx_filters)
        x = self.input


        #print(dL.shape,end='before')


        if(self.b_isLast):
            dL = dL.reshape(self.values_3.shape)

        else:
            dL = dL.reshape(self.result.shape)


        #print(dL.shape,end='dL')
        dU = self.__max__pool_backward(dL)
        #print(dU.shape,end='dU')
        dR = np.multiply(dU, self.v_relu_grad)



        dB = np.sum(dR,axis=(0,1),keepdims=True)
        #print(dB.shape)

        dW = np.zeros_like(w)
        #print(dW.shape,end='dW')

        f_dim = self.t_filter_size[0]

        #print(dR.shape,end='dr.shape\n')
        #print(x.shape, end='x.shape\n')
        #print(w.shape,end='w.shape\n')

        w_x = x.shape[0]
        max_offset = w_x - w.shape[0] +1
        #print(max_offset,end='offset\n')


        for a in range(f_dim):
            for b in range(f_dim):
                wX = x[a:(a+max_offset), b:(b+max_offset), :]
                #print(wX.shape,end='wX shape\n')
                dW[a,b,:] = np.sum(dR*wX,axis=(0,1,2))
        #print(dW.shape,end='dWshape\n')

        dX = np.zeros_like(x)

        Zpad = f_dim-1,f_dim-1

        dZpad = np.pad(dR, (Zpad,Zpad,(0,0)), 'constant', constant_values=0)

        #print(dR.shape,end='before pad \n')
        #print(dZpad.shape, end='after pad\n')


        for f in range(w.shape[3]): # f filter nr
            for c in range(x.shape[2]): # c channel
                dX[:,:,c] += self.__convolve2d__(dZpad[:,:,f], w[:,:,c,f])
        #print(dX.shape,end='dX.shape\n')
        #print(self.input.shape, end='input\n')

        return dW, dB, dX

    def update(self, dW, dB, learningRate):
        dW = dW.reshape(self.mx_filters.shape)
        dB = dB.reshape(self.v_bias.shape)
        self.mx_filters += learningRate*dW
        self.v_bias += learningRate*dB


    def __init_random_filters__(self):
        self.mx_filters = np.random.normal(Conv_layer.gauss_MEAN, Conv_layer.gauss_ST_DEVIATION, self.t_Allfilters_size)
        self.v_bias = np.random.normal(Conv_layer.gauss_MEAN, Conv_layer.gauss_ST_DEVIATION, self.i_num_of_filters)

    def __load_weights__(self,filter,bias):
        x = np.shape(filter)
        c = np.shape(bias)

        flag1 = ( x == self.t_Allfilters_size)
        flag2 = ( c == self.i_num_of_filters)

        if( (not flag1) or (not flag2) ):
            raise Exception('Conv_layer Exception: Wrong weights size!')


        else:
            self.mx_filters = filter
            self.v_bias = bias

    def __cnn_layer__(self, image):

        num_channels = np.shape(image)[2]

        conved_features = np.zeros((self.t_conv_size[0], self.t_conv_size[1], self.i_num_of_filters))

        for filter_i in range(self.i_num_of_filters):
            conved_img = np.zeros(self.t_conv_size)
            for channel_i in range(num_channels):
                filter = self.mx_filters[:,:,channel_i,filter_i]

                img = image[:,:,channel_i]
                conved_img += self.__convolve2d__(img,filter,self.b_padding)
            conved_img += self.v_bias[filter_i]
            conved_features[:,:,filter_i] = conved_img
        return conved_features


    """ DEPRECATED xD
    def __convolve__(self, image):
        features = np.array((self.t_conv_size[0], self.t_conv_size[1], self.t_conv_size[2], self.i_num_of_filters))

        for i in range(self.i_num_of_filters):
            features[:,:,i] = self.__convolve2d__(image, self.mx_filters[:,:,i])
            features[:,:,i] += self.v_bias[i]
        return features
    """

    def __convolve2d__(self, image, filter, padding = False):
        image_dim = np.array(np.shape(image))
        filter_dim = np.array(np.shape(filter))
        """Deprecated
        if image_dim  != (self.i_input_img_size, self.i_input_img_size):
            print('Conv_layer fail: Given input image size is not equal to one declared previously')
            return
        if filter_dim != (self.i_filter_size, self.i_filter_size):
            print('Conv_layer fail: Given filter size is not equal to one declared previously')
            return
        """
        if padding :
            target_dim = image_dim + filter_dim -1
        else:
            target_dim = image_dim - filter_dim +1
        fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(filter, target_dim)
        target = np.fft.ifft2(fft_result).real

        return target





    def __max__pooling__(self, features):
        pooling_dim = self.i_pooling_dim
        conv_dim, _, nb_features = np.shape(features)
        res_dim = int(conv_dim / pooling_dim)  # assumed square shape
        pooled_features = np.zeros((res_dim, res_dim, nb_features))

        self.mx_max_pool_map = np.zeros(features.shape)

        for feature_i in range(nb_features):
            for pool_row in range(res_dim):
                row_start = pool_row * pooling_dim
                row_end = row_start + pooling_dim

                for pool_col in range(res_dim):
                    col_start = pool_col * pooling_dim
                    col_end = col_start + pooling_dim

                    patch = features[row_start: row_end, col_start: col_end,feature_i]
                    x, y = np.unravel_index(np.argmax(patch,axis=None),patch.shape)
                    self.mx_max_pool_map[x+row_start, y+col_start, feature_i] = 1
                    pooled_features[pool_row, pool_col,feature_i] = np.max(patch)
        return pooled_features

    def __max__pool_backward(self, dX):

        pool_values = np.repeat( np.repeat(dX, self.i_pooling_dim, axis=0), self.i_pooling_dim, axis=1 )
        map_pool = self.mx_max_pool_map

        if(map_pool.shape[0] > pool_values.shape[0]):
            pool_values  = np.pad(pool_values, ((0,1), (0,1), (0,0)), 'constant', constant_values=0)

        return pool_values * map_pool

    def __move_to_3D__(self, image):

        if(len(image.shape) != 3):
            image = np.reshape(image, (image.shape[0], image.shape[0], 1))
        return image
