import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import ndimage




def myimshow(I, **kwargs):
    # utility function to show image
    plt.figure();
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)




with (open('../train.pkl','rb')) as f:
    data = pickle.load(f)


print(data[0].shape)

arr = np.array(data[0])

arr = arr

img = np.reshape(arr[72],(36,36))

myimshow(img)

plt.show()






