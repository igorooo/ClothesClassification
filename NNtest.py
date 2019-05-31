import numpy as np
import FullyConnectedLayer as fl
import pickle

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

sig = np.vectorize(sigmoid)

def extractData():
    with (open('../train.pkl', 'rb')) as f:
        data = pickle.load(f)

    images = np.array(data[0])
    images = images.reshape((images.shape[0], 36, 36, 1))
    labels_ = np.array(data[1])
    labels = np.zeros((labels_.shape[0], 10))

    for i in range(labels.shape[0]):
        labels[i, labels_[i]] = 1

    training_set = []

    tr_set_num = 55000

    tr_imgs = images[:tr_set_num, :,:,0]
    tr_labels = labels[:tr_set_num]

    training_set.append(tr_imgs)
    training_set.append(tr_labels)

    valid_set = []

    valid_imgs = images[tr_set_num:, :,:,0]
    valid_labels = labels[tr_set_num:]

    valid_set.append(valid_imgs)
    valid_set.append(valid_labels)

    return training_set,valid_set


tr,vl = extractData()



ff1 = fl.FF_layer(1296, 648, 0.2)
ff1.__init_random_weights__()

ff2 = fl.FF_layer(648, 324, 0.2)
ff2.__init_random_weights__()

ff3 = fl.FF_layer(324, 162, 0.2)
ff3.__init_random_weights__()

ff4 = fl.FF_layer(162, 10, 0, True)
ff4.__init_random_weights__()

def forwardPass(img):
    X = ff1.forwardPass(img)
    X = sig(X)
    X = ff2.forwardPass(X)
    X = sig(X)
    X = ff3.forwardPass(X)
    X = sig(X)
    X = ff4.forwardPass(X)
    return X


def learn(training_set, epochs=5000, learning_rate=0.01):
    x_tr_set = training_set[0].shape[0]




    for ep in range(epochs):

        img = training_set[0][ep,:,:]
        label = training_set[1][ep]

        convImg = img.flatten()
        img = convImg.reshape(convImg.shape[0])

        X = forwardPass(img)

        res = __classify_vector__(X)

        loss = __cross_entropy__(label, res)

        dX = ff4.backwardPass(loss)
        ff4.update(dX[0],dX[1], learning_rate)

        dX = ff3.backwardPass(sig(dX[2],True))
        ff3.update(dX[0], dX[1], learning_rate)

        dX = ff2.backwardPass(sig(dX[2],True))
        ff2.update(dX[0], dX[1], learning_rate)

        dX = ff1.backwardPass(sig(dX[2],True))
        ff1.update(dX[0], dX[1], learning_rate)




def vaidation(valid_set):
    x_valid_set = valid_set[0].shape[0]

    positive = 0

    epochs = 200

    for ep in range(epochs):

        img = valid_set[0][ep,:,:]
        label = valid_set[1][ep]

        convImg = img.flatten()
        img = convImg.reshape(convImg.shape[0])

        X = forwardPass(img)

        result = __classify__(X)


        if (result == np.argmax(label)):
            positive += 1

    return positive / epochs

def __classify__(x):
    return np.argmax(x,axis=-1)

def __classify_vector__(x):
    res = np.zeros((10))
    res[np.argmax(x,axis=-1)] = 1
    return res


def __cross_entropy__(label, y):
    label = label.reshape(y.shape)
    div = y - label
    return np.reshape(div,(y.shape[0], 1))





tr_set,val_set = extractData()

learn(tr_set)

result = vaidation(val_set)
print(result)