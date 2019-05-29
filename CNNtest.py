import CNN
import numpy as np


cnn = CNN.CNN(padding=(False,False))

cnn.__init_cnn_layers__()
cnn.__init_ff_layers__()
test = np.ones((36,36))


res = cnn.forwardPass(test)

print(res)