import time

import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

import pickle


TRAIN_SIZE = 1000000


def tf_idf(x_data):
    vectorizer = TfidfVectorizer()
    x_vec = vectorizer.fit_transform(x_data)
    return x_vec


def print_evaluation_scores(y_val, predicted):
    acc = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("accuracy:", acc)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)


# Generate a data set.
dataPath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/svm/data.txt'
labelPath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/svm/label.txt'
# train_data, train_labels, test_data, test_labels = read_dataset(featurePath, labelPath)
print('----------------reading----------------')
start_time = time.time()
input_data = open(dataPath)
vec = tf_idf(input_data)
labels = np.loadtxt(labelPath, dtype=float)
labels_vec = label_binarize(labels, classes=list(range(8)))
train_vec = vec[:TRAIN_SIZE]
test_vec = vec[TRAIN_SIZE:]
train_label = labels_vec[:TRAIN_SIZE]
test_label = labels_vec[TRAIN_SIZE:]
elapsed_time = time.time() - start_time
print('read data time: ', elapsed_time, 's')


print('----------------training----------------')
start_time2 = time.time()
model = OneVsRestClassifier(svm.SVC(kernel='linear'))
# OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_data, train_labels).predict(test_data)
clf = model.fit(train_vec, train_label)
elapsed_time2 = time.time() - start_time2
print('training time: ', elapsed_time2, 's')

s = pickle.dumps(model)
f = open('/data/users/lzh/bwu/model/svm/svm_linear_1000000_2.model', "wb+")
f.write(s)
f.close()
print("Model Saving Done\n")

train_predict = model.predict(train_vec)
test_predict = model.predict(test_vec)

print('The result of SVM is:')
print('----------------train-result----------------')
print_evaluation_scores(train_label, train_predict)
print('----------------test-result----------------')
# accuracy = accuracy_score(test_label, test_predict)
# print('accuracy: ' + str(accuracy))
print_evaluation_scores(test_label, test_predict)
