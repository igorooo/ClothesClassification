import numpy as np

def __max__pooling__(features, pooling_dim):


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

def __convolve2d__(image, filter, b = True):
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
    if b :
        target_dim = image_dim + filter_dim -1
    else:
        target_dim = image_dim - filter_dim +1

    fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(filter, target_dim)
    target = np.fft.ifft2(fft_result).real

    return target


test = np.ones((5,5))
filter = np.array([[0,0,0],[0,2,0],[0,0,0]])

testpool = np.ones((5,5,1))

#print(test)
#print(filter)

print(__max__pooling__(testpool,2).shape)
print('#######')
print(__convolve2d__(test,filter,False))
