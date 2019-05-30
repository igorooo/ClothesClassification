import numpy as np
import ConvLayer as CL
import FullyConnectedLayer as FFL
import Layer as L

cl = CL.Conv_layer((32,32,1),(3,3,1), 3)

cl.__init_random_filters__()

img = np.ones((32,32,1))

res = cl.forwardPass(img)

res = res.flatten()
#print(res.shape[0])

ffl = FFL.FF_layer(res.shape[0], 3, 0.25)
ffl.__init_random_weights__()

res2 = ffl.forwardPass(res)

dL = np.ones((3,1))

fflBres = ffl.backwardPass(dL,0.1)



cclBres = cl.backwardPass(fflBres[2],0.1)
print(cclBres)

#print(res2.shape)



test = np.array([[1,1,1],[2,2,2],[3,3,3]])
test2 = np.ones((3,5))

#print(test2.shape)

test1 = np.array([1,2,3])
#print(test1.shape)

#print(np.multiply(test2, test1))


    

    

"""

res = ffl.softmax(ffl.result)

print('#####')

print(res)

np.set_printoptions(suppress=True)
print(res)

"""


