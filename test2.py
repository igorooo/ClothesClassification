import numpy as np
import ConvLayer as CL
import FullyConnectedLayer as FFL
import Layer as L

cl = CL.Conv_layer((32,32,1),(3,3,1), 3)

cl.__init_random_filters__()

img = np.ones((32,32,1))

res = cl.forwardPass(img)

res = res.flatten()

ffl = FFL.FF_layer(res.shape[0], 3, 0.25)
ffl.__init_random_weights__()

res2 = ffl.forwardPass(res)

#print(res2.shape)



a = [cl, ffl]
i = 0
for layer in a:
    print(i)
    i +=1


    

    

"""

res = ffl.softmax(ffl.result)

print('#####')

print(res)

np.set_printoptions(suppress=True)
print(res)

"""


