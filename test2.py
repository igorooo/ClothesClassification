import numpy as np


D3 = np.zeros((3,3,0))

print(D3)

D2 = np.ones((3,3))

print(D2)

D3 = np.dstack((D3,D2))

print(D3.shape)
print(D3)

print('####')
print(D3[:,:,0])

D3_2 = D3[:,:,0]
print(D3_2)
