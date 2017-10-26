import sklearn.externals.joblib as joblib
import time
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_files
from sklearn.utils import shuffle
mem = Memory("./mycache")


@mem.cache
def get_data(type):
    return load_svmlight_files(("../data/Fold1/" + type, "../data/Fold2/" + type, "../data/Fold3/" +
                                type, "../data/Fold4/" + type, "../data/Fold5/" + type))


data = get_data('train.txt')

new_data = [[], []]
fl = [0, 0, 0, 0, 0]
features_to_skip = []
amount = 30000

for i in range(5):
    for j in range(len(data[i * 2 + 1])):
        if fl[int(data[i * 2 + 1][j])] < amount:
            new_data[0].append([])
            for k in range(136):
                if 104 < k < 128 or k > 128 or k % 5 == 1 or k % 5 == 0:
                    continue
                new_data[0][len(new_data[0]) - 1].append(data[i * 2][j, k])
            if data[i * 2 + 1][j] < 2:
                new_data[1].append(0)
                fl[int(data[i * 2 + 1][j])] += 1
            else:
                new_data[1].append(1)
                fl[int(data[i * 2 + 1][j])] += 1.5
        if fl[0] > amount and fl[1] > amount and fl[2] > amount and fl[3] > amount and fl[4] > amount:
            break

new_data[0], new_data[1] = shuffle(new_data[0], new_data[1], random_state=0)
np.save('data4.txt', new_data)

