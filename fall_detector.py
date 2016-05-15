import numpy as np
import math
from collections import deque
from sklearn import svm
from sklearn.externals import joblib
clf = svm.SVC()

queue = deque([])

vectors = []
labels = []

# history = np.zeros((20,2))
index = 0

def pre_process(sequence):
    FA = sequence[19][0]
    AR = sequence[19][1]
    HP = sequence[19][2]
    MC = sequence[19][3]
    Dx = math.fabs(HP[1] - MC[1])
    LEN = math.sqrt(math.pow(HP[0] - queue[0][2][0], 2) + math.pow(HP[1] - queue[0][2][1], 2))
    return [FA, AR, Dx, LEN]

def update_params(FA, AR, HP, MC, label, reset):
    global index
    if reset:
        queue.clear()

    queue.append((FA, AR, HP, MC))
    if len(queue) > 20:
        queue.popleft()
        # arr = np.array(queue)
        # vectors.append((FA, AR, math.fabs(HP[1] - MC[1]), math.sqrt(math.pow(HP[0] - queue[0][2][0], 2) + math.pow(HP[1] - queue[0][2][1], 2))))
        vectors.append(pre_process(queue))
        if label:
            labels.append(1)
        else:
            labels.append(-1)
        # print(vectors)


def fall_detector_train():
    train_data = vectors
    global clf
    # global train_data
    # nsamples, nx, ny = train_data.shape
    # train_data = train_data.reshape((nsamples,nx*ny))
    clf.fit(train_data, labels) #,class_weight={1:10,-1:1}
    joblib.dump(clf, 'fall_detector_model.pkl')


def load_model():
    global clf
    clf = joblib.load('fall_detector_model.pkl')


def predict(FA, AR, HP, MC, reset):
    if reset:
        queue.clear()

    queue.append((FA, AR, HP, MC))
    if len(queue) == 20:
        # arr = np.array(queue)
        # feature = arr.flatten()
        feature = pre_process(queue)
        queue.popleft()
        # print(feature)
        if clf is None:
            load_model()
        if clf is None:
            return None
        else:
            if (feature[0] > 120) & (feature[1] < 45) & (feature[2] > 0.6) & (feature[3] > 70):
                return 1
                print(feature)
            else:
                return -1
            # return clf.predict([feature])
    else:
        return None

