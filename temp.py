import numpy as np

arr = np.ones((14,14,8))

print(arr[:,:,0])


arrr = np.pad(arr, ((0,1), (0,1), (0,0)), 'constant', constant_values=0)

print(arrr.shape)
print(arrr[:,:,0])