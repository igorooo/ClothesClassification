import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import ndimage


def convolve2d(image, feature):
    image_dim = np.array(image.shape)
    feature_dim = np.array(feature.shape)
    target_dim = image_dim + feature_dim - 1
    fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim)
    target = np.fft.ifft2(fft_result).real

    return target


def __convolve2d__(image, filter,b=True):
    image_dim = np.shape(image)
    filter_dim = np.shape(filter)


    image_dim = np.array(image_dim)
    filter_dim = np.array(filter_dim)

    if b:
        target_dim = image_dim + filter_dim - 1
    else:
        target_dim = image_dim - filter_dim + 1
    fft_result = np.fft.fft2(image, target_dim) * np.fft.fft2(filter, target_dim)
    target = np.fft.ifft2(fft_result).real

    return target

def __max__pooling__(features):
    pooling_dim = 2

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




def myimshow(I, **kwargs):
    # utility function to show image
    plt.figure();
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)

def genSinusoid(sz, A, omega, rho):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1)) # a BUG is fixed in this line

    stimuli = A * np.cos(omega[0] * x  + omega[1] * y + rho)
    return stimuli


def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0] /2 ), int(sz[1]/2 ))
    [x, y] = np.meshgrid(range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = omega ** 2 / (4 * np.pi * K ** 2) * np.exp(- omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    #     myimshow(gauss)
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    #     myimshow(sinusoid)

    gabor = gauss * sinusoid
    return gabor


#def gabor_kernel()


gf = genGabor((36,36),3.5, np.pi/3 , func=np.cos)


#myimshow(gf)


with (open('../train.pkl','rb')) as f:
    data = pickle.load(f)

arr = np.asarray(data[0])

arr = arr

img = np.reshape(arr[72],(36,36))

myimshow(img)


filter = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
filter2 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
filter3 = np.array([[-1,0,1],
                    [0,1,0],
                    [1,0,-1]])

filter4 = np.array([[1,0,-1],
                    [0,1,0],
                    [-1,0,1]])



img1 = ndimage.convolve(img,filter,mode='constant',cval=0.0)
img2 = ndimage.convolve(img,filter2,mode='constant',cval=0.0)
img3 = ndimage.convolve(img,filter3,mode='constant',cval=0.0)
img4 = ndimage.convolve(img,filter4,mode='constant',cval=0.0)

"""
myimshow(img1)
myimshow(img2)
myimshow(img3)
"""
#myimshow(img4)

img = np.dstack((img,img))
print(img.shape)

filter4 = np.dstack((filter4,filter4))
print(filter4.shape)

imgFFT = convolve2d(img, filter4)

#imgFFT = np.fft.fft2(img, img.shape) * np.fft.fft2(filter4, img.shape)
#imgFFT = np.flip(np.fft.fft2(imgFFT).real)

myimshow(imgFFT)

imgConvFull = __convolve2d__(img,filter4)
imgConNotFul = __convolve2d__(img,filter4,False)

#myimshow(imgConvFull)
#myimshow(imgConNotFul)

PoolImg = np.zeros((1,36,36))
print(img.shape)
print(img[0,0])
print(PoolImg[0,0,0])

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        PoolImg[0,i,j] = img[i,j]


print(np.shape(PoolImg))


poolRes = __max__pooling__(PoolImg)[0]

#myimshow(poolRes)



print(img.shape)
print(imgConvFull.shape)
print(imgConNotFul.shape)

plt.show()






