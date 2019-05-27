import numpy as np

class Conv_layer():

    gauss_MEAN = 0
    gauss_ST_DEVIATION = 1


    def __init__(self, input_img_size, filter_size, num_of_filters, pooling_dim = 2, padding = True):
        self.t_input_img_size = np.array(input_img_size)  #delete
        self.t_filter_size = np.array(filter_size)
        self.i_num_of_filters = num_of_filters
        self.i_pooling_dim = pooling_dim
        self.b_padding = padding

        if self.b_padding:
            self.t_conv_size = (self.t_input_img_size[0] + self.t_filter_size[0] -1,self.t_input_img_size[0] + self.t_filter_size[0] -1)

        if not self.b_padding:
            self.t_conv_size = (self.t_input_img_size[0] - self.t_filter_size[0] +1,self.t_input_img_size[0] - self.t_filter_size[0] +1)

        self.t_Allfilters_size = (self.t_filter_size[0], self.t_filter_size[1], self.t_filter_size[2], self.i_num_of_filters)


        self.mx_filters = []
        self.v_bias = []

        self.features = []



    def forwardPass(self, image):
        if len(self.mx_filters) == 0 or len(self.v_bias) == 0:
            raise Exception('Conv_layer Exception: Weights non initialized!')
            return

        features = self.__cnn_layer__(image)
        features = self.__relu__(features)
        features = self.__max__pooling__(features)

        return features

    def checkResultSize(self):
        return np.shape(self.forwardPass(np.zeros(self.t_input_img_size)))


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
                conved_img += self.__convolve2d__(img,filter)
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

    def __convolve2d__(self, image, filter):
        image_dim = np.shape(image)
        filter_dim = np.shape(filter)
        """Deprecated
        if image_dim  != (self.i_input_img_size, self.i_input_img_size):
            print('Conv_layer fail: Given input image size is not equal to one declared previously')
            return
        if filter_dim != (self.i_filter_size, self.i_filter_size):
            print('Conv_layer fail: Given filter size is not equal to one declared previously')
            return
        """

        target_dim = self.t_conv_size
        fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(filter, target_dim)
        target = np.fft.ifft2(fft_result).real

        return target

    def __relu__(self,x):
        z = np.zeros_like(x)
        return np.where(x > z, x, z)



    def __max__pooling__(self, features):
        pooling_dim = self.i_pooling_dim

        conv_dim, _, nb_features = np.shape(features)
        res_dim = int(conv_dim / pooling_dim)  # assumed square shape

        pooled_features = np.zeros((res_dim, res_dim, nb_features))

        for feature_i in range(nb_features):
            for pool_row in range(res_dim):
                row_start = pool_row * pooling_dim
                row_end = row_start + pooling_dim

                for pool_col in range(res_dim):
                    col_start = pool_col * pooling_dim
                    col_end = col_start + pooling_dim

                    patch = features[row_start: row_end, col_start: col_end,feature_i]
                    pooled_features[pool_row, pool_col,feature_i] = np.max(patch)
        return pooled_features
