import numpy as np

class Conv_layer():

    gauss_MEAN = 0
    gauss_ST_DEVIATION = 1


    def __init__(self, input_img_size, filter_size, num_of_filters, pooling_dim, stride = 1, padding = True):
        self.i_input_img_size = input_img_size  #delete
        self.i_filter_size = filter_size
        self.i_num_of_filters = num_of_filters
        self.i_pooling_dim = pooling_dim
        self.i_stride = stride
        self.b_padding = padding

        if self.b_padding:
            self.i_conv_size = self.i_input_img_size + self.i_filter_size -1

        if not self.b_padding:
            self.i_conv_size = self.i_input_img_size - self.i_filter_size + 1


        self.i_conv_size = (self.i_input_img_size+2*self.i_padding-self.i_filter_size) / (self.i_stride + 1)
        self.mx_filters = []
        self.v_bias = []

        self.mx_partial_results = []


    def preceed

    def __init_random_filters__(self):
        self.mx_filters = np.random.normal(Conv_layer.gauss_MEAN, Conv_layer.gauss_ST_DEVIATION, (self.i_conv_size, self.i_conv_size, self.i_num_of_filters))
        self.v_bias = np.random.normal(Conv_layer.gauss_MEAN, Conv_layer.gauss_ST_DEVIATION, self.i_num_of_filters)

    def __load_weights__(self,filter,bias):
        x = np.shape(filter)
        c = np.shape(bias)

        flag1 = ( x == (self.i_filter_size, self.i_filter_size, self.i_num_of_filters))
        flag2 = ( c == self.i_num_of_filters)

        if( (not flag1) or (not flag2) ):
            raise Exception('Conv_layer Exception: Wrong weights size!')

        else:
            self.mx_filters = filter
            self.v_bias = bias

    def __convolve2d__(self, image, filter):
        image_dim = np.shape(image)
        filter_dim = np.shape(filter)
        if image_dim  != (self.i_input_img_size, self.i_input_img_size):
            print('Conv_layer fail: Given input image size is not equal to one declared previously')
            return
        if filter_dim != (self.i_filter_size, self.i_filter_size):
            print('Conv_layer fail: Given filter size is not equal to one declared previously')
            return

        image_dim = np.array(image_dim)
        filter_dim = np.array(filter_dim)

        target_dim = self.i_conv_size
        fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(filter, target_dim)
        target = np.fft.ifft2(fft_result).real

        return target

    def __relu__(self,x):
        z = np.zeros_like(x)
        return np.where(x > z, x, z)



    def __max__pooling__(self, features):
        pooling_dim = self.i_pooling_dim

        nb_features, conv_dim, _ = np.shape(features)
        res_dim = int(conv_dim / pooling_dim)  # assumed square shape

        pooled_features = np.zeros((nb_features, res_dim, res_dim))

        for feature_i in range(nb_features):
            for pool_row in range(res_dim):
                row_start = pool_row * pooling_dim
                row_end = row_start + pooling_dim

                for pool_col in range(res_dim):
                    col_start = pool_col * pooling_dim
                    col_end = col_start + pooling_dim

                    patch = features[feature_i, row_start: row_end, col_start: col_end]
                    pooled_features[feature_i, pool_row, pool_col] = np.max(patch)
        return pooled_features























