""" Deep Neural Network for dataset classification task.
"""
from __future__ import division, print_function, absolute_import

import os
import time

import numpy as np
import tflearn
from sklearn.metrics import accuracy_score, f1_score
from tflearn.data_utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

# CLASS_NUMBER = 8
EPOCH_SIZE = 300
BATCH_SIZE = 32

TRAIN_SIZE = 990000
VALIDATION_SIZE = 10000
TENSORBOARD_DIR = "/data/users/lzh/bwu/model/net/section/tflearn_logs_5/"


def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("accuracy:", accuracy)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)


# Data loading and preprocessing
start_time = time.time()
features = np.load("/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data-bert.npy").astype(dtype=np.float32)
labels = np.load("/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/label.npy").astype(dtype=np.int32)
labels = to_categorical(labels, nb_classes=8)

# Generate a validation set.
train_data = features[:TRAIN_SIZE]
train_labels = labels[:TRAIN_SIZE]
validation_data = features[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
validation_labels = labels[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
test_data = features[TRAIN_SIZE+VALIDATION_SIZE:]
test_labels = labels[TRAIN_SIZE+VALIDATION_SIZE:]
print('Dataset setting is finished!')

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 768])  # set input data's shape
dense1 = tflearn.fully_connected(input_layer, 64, activation='relu',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.5)
dense2 = tflearn.fully_connected(dropout1, 8, activation='softmax')
net = tflearn.regression(dense2, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

# Training
print('--------Training Start--------')
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=TENSORBOARD_DIR,
                    checkpoint_path='/data/users/lzh/bwu/model/net/checkpoints_5/')
model.fit(train_data, train_labels, n_epoch=EPOCH_SIZE,
          validation_set=(validation_data, validation_labels),
          show_metric=True, batch_size=BATCH_SIZE, run_id="framework_one_model_5")
print('--------Training End--------')

print('--------Model Saving Start--------')
model.save("/data/users/lzh/bwu/model/net/section_5.tfl")
print('--------Model Saving End--------')

print('--------Evaluation Start--------')
# model.load("/data/users/lzh/bwu/model/net/section.tfl")
test_predict = model.predict(test_data)
print('Model test is finished')
test_predict_trans = [np.argmax(one_hot) for one_hot in test_predict]
test_labels_trans = [np.argmax(one_hot) for one_hot in test_labels]
print_evaluation_scores(test_labels_trans, test_predict_trans)
print('--------Evaluation End--------')

print("end......")
