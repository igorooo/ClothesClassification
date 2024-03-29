import numpy as np
import pickle
import CNN
from matplotlib import pyplot as plt

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

    valid_imgs = images[tr_set_num:, :]
    valid_labels = labels[tr_set_num:]

    valid_set.append(valid_imgs)
    valid_set.append(valid_labels)

    return training_set,valid_set


tr,vl = extractData()





cnn = CNN.CNN()

cnn.init_random()

cnn.learn(tr)


res = cnn.vaidation(vl)

print(res)







