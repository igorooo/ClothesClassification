import numpy as np
import ConvLayer as CL

cl = CL.Conv_layer((32,32,1),(3,3,1), 3)

cl.__init_random_filters__()

img = np.ones((32,32,1))

print(cl.forwardPass(img).shape)

print(cl.checkResultSize())

