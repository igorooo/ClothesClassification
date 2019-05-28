import numpy as np



test = np.zeros((3,3,1))

print(test.shape)
print(len(test.shape))

test2 = np.ones((3,3))

test2 = np.reshape(test2, (3,3,1))

print(test2)
print(test2.shape)
