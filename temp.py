import numpy as np

def __max__pooling__(features, pooling_dim):


    conv_dim, _, nb_features = np.shape(features)
    res_dim = int(conv_dim / pooling_dim)  # assumed square shape

    mx_max_pool_map = np.zeros(features.shape)

    pooled_features = np.zeros((res_dim, res_dim, nb_features))

    for feature_i in range(nb_features):
        for pool_row in range(res_dim):
            row_start = pool_row * pooling_dim
            row_end = row_start + pooling_dim

            for pool_col in range(res_dim):
                col_start = pool_col * pooling_dim
                col_end = col_start + pooling_dim

                patch = features[row_start: row_end, col_start: col_end,feature_i]
                x, y = np.unravel_index(np.argmax(patch, axis=None), patch.shape)
                mx_max_pool_map[x+row_start, y+col_start, feature_i] = 1
                pooled_features[pool_row, pool_col,feature_i] = np.max(patch)
    return pooled_features, mx_max_pool_map

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

testpool = np.arange(16).reshape((4,4,1))

testpool[0,0,0] = 20
testpool[2,1,0] = 21

print(testpool)
print('^^^')

maxpool, map = __max__pooling__(testpool,2)

print(maxpool[:,:,0])
print('#######')
print(map[:,:,0])
#print(__convolve2d__(test,filter,False))

print('$$$$$$')

testpool = np.arange(4).reshape((2,2,1))
print(np.repeat(np.repeat(testpool,4, axis=0),4,axis=1).shape)

