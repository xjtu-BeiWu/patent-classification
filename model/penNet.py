import os
import time

import numpy as np
import tensorflow as tf
import tflearn
from sklearn.metrics import accuracy_score, f1_score
from tflearn.data_utils import to_categorical, shuffle
from tflearn.callbacks import Callback

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# tf.reset_default_graph()
TRAIN_SIZE = 1350000
VALIDATION_SIZE = 20000

# numbers of categories
# SUBG_NUM = 129574
G_NUM = 6746
SUBC_NUM = 627
C_NUM = 123
SUBS_NUM = 27
SEC_NUM = 8

Citation_Dim = 128


# Data loading and preprocessing
def load_data(feature_path1, label_path1):
    start_time = time.time()
    features = np.load(feature_path1).astype(dtype=np.float32)
    # features = np.load("/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data-bert.npy").astype(dtype=np.float32)
    labels = np.load(label_path1).astype(dtype=np.int32)
    # labels = np.load("/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/label.npy").astype(dtype=np.int32)
    # subG_mask_labels = np.load("").astype(dtype=np.int32)
    group_mask_labels = np.load("").astype(dtype=np.int32)
    subC_mask_labels = np.load("").astype(dtype=np.int32)
    class_mask_labels = np.load("").astype(dtype=np.int32)
    subS_mask_labels = np.load("").astype(dtype=np.int32)

    labels = to_categorical(labels, nb_classes=8)

    # Generate a validation set.
    train_data = features[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]
    # train_subG_mask_labels = subG_mask_labels[:TRAIN_SIZE]
    train_group_mask_labels = group_mask_labels[:TRAIN_SIZE]
    train_subC_labels = subC_mask_labels[:TRAIN_SIZE]
    train_class_labels = class_mask_labels[:TRAIN_SIZE]
    train_subS_labels = subS_mask_labels[:TRAIN_SIZE]

    validation_data = features[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_labels = labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    # validation_subG_labels = subG_mask_labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_group_labels = group_mask_labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_subC_labels = subC_mask_labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_class_labels = class_mask_labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_subS_labels = subS_mask_labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]

    test_data = features[TRAIN_SIZE + VALIDATION_SIZE:]
    test_labels = labels[TRAIN_SIZE + VALIDATION_SIZE:]
    # test_subG_labels = subG_mask_labels[TRAIN_SIZE + VALIDATION_SIZE:]
    test_group_labels = group_mask_labels[TRAIN_SIZE + VALIDATION_SIZE:]
    test_subC_labels = subC_mask_labels[TRAIN_SIZE + VALIDATION_SIZE:]
    test_class_labels = class_mask_labels[TRAIN_SIZE + VALIDATION_SIZE:]
    test_subS_labels = subS_mask_labels[TRAIN_SIZE + VALIDATION_SIZE:]
    elapsed_time = time.time() - start_time
    print('Dataset setting is finished! Time cost is: ', elapsed_time, 's')
    return train_data, train_labels, train_group_mask_labels, train_subC_labels, train_class_labels, \
           train_subS_labels, \
           validation_data, validation_labels, validation_group_labels, validation_subC_labels, \
           validation_class_labels, validation_subS_labels, \
           test_data, test_labels, test_group_labels, test_subC_labels, test_class_labels, test_subS_labels


def load_data2(data_path, pre_label_path):
    start_time = time.time()
    features = np.load(data_path).astype(dtype=np.float32)
    labels = np.load(pre_label_path).astype(dtype=np.int32)
    labels = to_categorical(labels, nb_classes=8)

    # Generate a validation set.
    train_data = features[:TRAIN_SIZE]
    train_label = labels[:TRAIN_SIZE]
    validation_data = features[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_label = labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]

    test_data = features[TRAIN_SIZE + VALIDATION_SIZE:]
    test_label = labels[TRAIN_SIZE + VALIDATION_SIZE:]
    elapsed_time = time.time() - start_time
    print('Dataset setting is finished! Time cost is: ', elapsed_time, 's')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def load_data3(data_path, pre_label_path):
    start_time = time.time()
    # features = np.load(data_path).astype(dtype=np.int32)
    features = np.load(data_path).astype(dtype=np.float32)
    labels = np.load(pre_label_path).astype(dtype=np.int32)
    labels = to_categorical(labels, nb_classes=8)

    # Generate a validation set.
    train_data = features[:TRAIN_SIZE]
    train_label = labels[:TRAIN_SIZE]
    validation_data = features[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_label = labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]

    test_data = features[TRAIN_SIZE + VALIDATION_SIZE:]
    test_label = labels[TRAIN_SIZE + VALIDATION_SIZE:]
    elapsed_time = time.time() - start_time
    print('Dataset setting is finished! Time cost is: ', elapsed_time, 's')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def load_data4(data_path, pre_label_path):
    start_time = time.time()
    # features = np.load(data_path).astype(dtype=np.int32)
    features = np.load(data_path).astype(dtype=np.float32)
    labels = np.load(pre_label_path).astype(dtype=np.int32)
    labels = to_categorical(labels, nb_classes=8)

    # Generate a validation set.
    train_data = features[:TRAIN_SIZE]
    train_label = labels[:TRAIN_SIZE]
    validation_data = features[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_label = labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]

    test_data = features[TRAIN_SIZE + VALIDATION_SIZE:]
    test_label = labels[TRAIN_SIZE + VALIDATION_SIZE:]
    elapsed_time = time.time() - start_time
    print('Dataset setting is finished! Time cost is: ', elapsed_time, 's')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def load_data_shuffle(data_path, pre_label_path):
    start_time = time.time()
    # features = np.load(data_path).astype(dtype=np.int32)
    features = np.load(data_path).astype(dtype=np.float32)
    labels = np.load(pre_label_path).astype(dtype=np.int32)
    labels = to_categorical(labels, nb_classes=8)

    # Generate a validation set.
    train_data = features[:TRAIN_SIZE]
    train_label = labels[:TRAIN_SIZE]
    validation_data = features[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_label = labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    test_data = features[TRAIN_SIZE + VALIDATION_SIZE:]
    test_label = labels[TRAIN_SIZE + VALIDATION_SIZE:]
    elapsed_time = time.time() - start_time
    print('Dataset setting is finished! Time cost is: ', elapsed_time, 's')
    start_time = time.time()
    data_train = [(t_d, t_l) for t_d, t_l in zip(train_data, train_label)]
    np.random.shuffle(data_train)
    # train_data = np.array([t_d for t_d, t_l in data_train]).astype(dtype=np.float32)
    # train_label = np.array([t_l for t_d, t_l in data_train]).astype(dtype=np.int32)
    train_data = [t_d for t_d, t_l in data_train]
    train_label = [t_l for t_d, t_l in data_train]
    elapsed_time = time.time() - start_time
    print('Dataset shuffle is finished! Time cost is: ', elapsed_time, 's')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def load_data_shuffle2(data_path, pre_label_path):
    start_time = time.time()
    # features = np.load(data_path).astype(dtype=np.int32)
    features = np.load(data_path).astype(dtype=np.float32)
    labels = np.load(pre_label_path).astype(dtype=np.int32)
    labels = to_categorical(labels, nb_classes=8)

    # start_time = time.time()
    data_train = [(d, l) for d, l in zip(features, labels)]
    np.random.shuffle(data_train)
    # train_data = np.array([t_d for t_d, t_l in data_train]).astype(dtype=np.float32)
    # train_label = np.array([t_l for t_d, t_l in data_train]).astype(dtype=np.int32)
    features = [d for d, l in data_train]
    labels = [l for d, l in data_train]
    elapsed_time = time.time() - start_time
    print('Dataset shuffle is finished! Time cost is: ', elapsed_time, 's')

    # Generate a validation set.
    train_data = features[:TRAIN_SIZE]
    train_label = labels[:TRAIN_SIZE]
    validation_data = features[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    validation_label = labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
    test_data = features[TRAIN_SIZE + VALIDATION_SIZE:]
    test_label = labels[TRAIN_SIZE + VALIDATION_SIZE:]
    elapsed_time = time.time() - start_time
    print('Dataset setting is finished! Time cost is: ', elapsed_time, 's')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def create_penNet(dim):
    X = tf.placeholder('float', [None, dim])
    # Y = tf.placeholder('int', [None, SEC_NUM])
    Z_0 = tf.placeholder('int', [None, G_NUM])
    Z_1 = tf.placeholder('int', [None, SUBC_NUM])
    Z_2 = tf.placeholder('int', [None, C_NUM])
    Z_3 = tf.placeholder('int', [None, SUBS_NUM])
    # network = tflearn.input_data(shape=[None, EMBEDDING_DIM])
    group_embedding = tf.Variable(tf.random_normal([G_NUM, dim]))
    subclass_embedding = tf.Variable(tf.random_normal([SUBC_NUM, dim]))
    class_embedding = tf.Variable(tf.random_normal([C_NUM, dim]))
    subsection_embedding = tf.Variable(tf.random_normal([SUBS_NUM, dim]))
    section_embedding = tf.Variable(tf.random_normal([SEC_NUM, dim]))
    # SG_output = _cat_weighted(X, subgroup_embedding, W_SG)
    G_output = _cat_weighted1(X, group_embedding, Z_0)
    SC_output = _cat_weighted1(G_output, subclass_embedding, Z_1)
    C_output = _cat_weighted1(SC_output, class_embedding, Z_2)
    SS_output = _cat_weighted1(C_output, subsection_embedding, Z_3)
    pro = tf.matmul(SS_output, tf.transpose(section_embedding))
    network = tflearn.regression(pro, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    return network


def create_penNet2(em_dim):
    # Building 'penNet'
    input_layer = tflearn.input_data(shape=[None, em_dim + G_NUM + SUBC_NUM + C_NUM + SUBS_NUM])
    input_p = input_layer[:, 0:em_dim]
    input_g = input_layer[:, em_dim:em_dim + G_NUM]
    input_sc = input_layer[:, em_dim + G_NUM:em_dim + G_NUM + SUBC_NUM]
    input_c = input_layer[:, em_dim + G_NUM + SUBC_NUM:em_dim + G_NUM + SUBC_NUM + C_NUM]
    input_ss = input_layer[:, em_dim + G_NUM + SUBC_NUM + C_NUM:]
    group_embedding = tf.Variable(tf.random_normal([G_NUM, em_dim]), trainable=True)
    subclass_embedding = tf.Variable(tf.random_normal([SUBC_NUM, em_dim]))
    class_embedding = tf.Variable(tf.random_normal([C_NUM, em_dim]))
    subsection_embedding = tf.Variable(tf.random_normal([SUBS_NUM, em_dim]))
    section_embedding = tf.Variable(tf.random_normal([SEC_NUM, em_dim]))
    # SG_output = _cat_weighted(X, subgroup_embedding, W_SG)
    G_output = _cat_weighted1(input_p, group_embedding, input_g)
    SC_output = _cat_weighted1(G_output, subclass_embedding, input_sc)
    C_output = _cat_weighted1(SC_output, class_embedding, input_c)
    SS_output = _cat_weighted1(C_output, subsection_embedding, input_ss)
    network = tf.matmul(SS_output, tf.transpose(section_embedding))
    network = tflearn.fully_connected(network, SEC_NUM, activation='softmax')  # not sure
    network = tflearn.regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    return network


def create_penNet3(in_dim, em_dim):
    # Building 'penNet'
    input_layer = tflearn.input_data(shape=[None, in_dim + G_NUM + SUBC_NUM + C_NUM + SUBS_NUM], name='input')
    input_p = input_layer[:, 0:in_dim]
    input_ss = input_layer[:, in_dim:in_dim + SUBS_NUM]
    input_c = input_layer[:, in_dim + SUBS_NUM:in_dim + SUBS_NUM + C_NUM]
    input_sc = input_layer[:, in_dim + SUBS_NUM + C_NUM:in_dim + SUBS_NUM + C_NUM + SUBC_NUM]
    input_g = input_layer[:, in_dim + SUBS_NUM + C_NUM + SUBC_NUM:]
    group_embedding = tf.Variable(tf.random_normal([G_NUM, em_dim]), trainable=True, name='group_embedding')
    subclass_embedding = tf.Variable(tf.random_normal([SUBC_NUM, em_dim]), name='subclass_embedding')
    class_embedding = tf.Variable(tf.random_normal([C_NUM, em_dim]), name='class_embedding')
    subsection_embedding = tf.Variable(tf.random_normal([SUBS_NUM, em_dim]), name='subsection_embedding')
    section_embedding = tf.Variable(tf.random_normal([SEC_NUM, em_dim]), name='section_embedding')
    # SG_output = _cat_weighted(X, subgroup_embedding, W_SG)
    network = tflearn.embedding(input_p, input_dim=110240, output_dim=128, name='word_embedding')
    network = tflearn.lstm(network, em_dim, dropout=0.8, name='lstm_weight')
    # network = tflearn.bidirectional_rnn(network, tflearn.BasicLSTMCell(64), tflearn.BasicLSTMCell(64),
    #                                     name='bilstm_weight')
    # network = tflearn.dropout(network, 0.8)
    G_output = _cat_weighted1(network, group_embedding, input_g)
    SC_output = _cat_weighted1(G_output, subclass_embedding, input_sc)
    C_output = _cat_weighted1(SC_output, class_embedding, input_c)
    SS_output = _cat_weighted1(C_output, subsection_embedding, input_ss)
    network = tf.matmul(SS_output, tf.transpose(section_embedding), name='section_weight')
    network = tflearn.fully_connected(network, SEC_NUM, activation='softmax')  # not sure
    network = tflearn.regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    return network


def create_penNet4(in_dim, em_dim):
    # Building 'penNet'
    input_layer = tflearn.input_data(shape=[None, in_dim + G_NUM + SUBC_NUM + C_NUM + SUBS_NUM], name='input')
    input_p = input_layer[:, 0:in_dim]
    input_ss = input_layer[:, in_dim:in_dim + SUBS_NUM]
    input_c = input_layer[:, in_dim + SUBS_NUM:in_dim + SUBS_NUM + C_NUM]
    input_sc = input_layer[:, in_dim + SUBS_NUM + C_NUM:in_dim + SUBS_NUM + C_NUM + SUBC_NUM]
    input_g = input_layer[:, in_dim + SUBS_NUM + C_NUM + SUBC_NUM:]
    group_embedding = tf.Variable(tf.random_normal([G_NUM, em_dim]), name='group_embedding')
    subclass_embedding = tf.Variable(tf.random_normal([SUBC_NUM, em_dim]), name='subclass_embedding')
    class_embedding = tf.Variable(tf.random_normal([C_NUM, em_dim]), name='class_embedding')
    subsection_embedding = tf.Variable(tf.random_normal([SUBS_NUM, em_dim]), name='subsection_embedding')
    section_embedding = tf.Variable(tf.random_normal([SEC_NUM, em_dim]), name='section_embedding')
    # SG_output = _cat_weighted(X, subgroup_embedding, W_SG)
    network = tflearn.embedding(input_p, input_dim=110240, output_dim=128, name='word_embedding')
    network = tflearn.lstm(network, em_dim, dropout=0.8, name='lstm_weight')
    # network = tflearn.bidirectional_rnn(network, tflearn.BasicLSTMCell(128), tflearn.BasicLSTMCell(128),
    #                                     name='bilstm_weight')
    # network = tflearn.dropout(network, 0.8)
    G_output = _cat_weighted1(network, group_embedding, input_g)
    SC_output = _cat_weighted1(G_output, subclass_embedding, input_sc)
    C_output = _cat_weighted1(SC_output, class_embedding, input_c)
    SS_output = _cat_weighted1(C_output, subsection_embedding, input_ss)
    network = tf.matmul(SS_output, tf.transpose(section_embedding), name='section_weight')
    # network = tflearn.fully_connected(network, SEC_NUM, activation='softmax')  # not sure
    network = tflearn.softmax(network)
    network = tflearn.regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    return network


def create_citation(in_dim, em_dim):
    # Building 'citation'
    input_layer = tflearn.input_data(shape=[None, in_dim + G_NUM + SUBC_NUM + C_NUM + SUBS_NUM], name='input')
    input_p = input_layer[:, 0:in_dim]
    input_ss = input_layer[:, in_dim:in_dim + SUBS_NUM]
    input_c = input_layer[:, in_dim + SUBS_NUM:in_dim + SUBS_NUM + C_NUM]
    input_sc = input_layer[:, in_dim + SUBS_NUM + C_NUM:in_dim + SUBS_NUM + C_NUM + SUBC_NUM]
    input_g = input_layer[:, in_dim + SUBS_NUM + C_NUM + SUBC_NUM:]
    group_embedding = tf.Variable(tf.random_normal([G_NUM, em_dim]), name='group_embedding')
    subclass_embedding = tf.Variable(tf.random_normal([SUBC_NUM, em_dim]), name='subclass_embedding')
    class_embedding = tf.Variable(tf.random_normal([C_NUM, em_dim]), name='class_embedding')
    subsection_embedding = tf.Variable(tf.random_normal([SUBS_NUM, em_dim]), name='subsection_embedding')
    section_embedding = tf.Variable(tf.random_normal([SEC_NUM, em_dim]), name='section_embedding')
    # SG_output = _cat_weighted(X, subgroup_embedding, W_SG)
    network = tflearn.embedding(input_p, input_dim=110240, output_dim=128, name='word_embedding')
    network = tflearn.lstm(network, em_dim, dropout=0.8, name='lstm_weight')
    G_output = _cat_weighted1(network, group_embedding, input_g)
    SC_output = _cat_weighted1(G_output, subclass_embedding, input_sc)
    C_output = _cat_weighted1(SC_output, class_embedding, input_c)
    SS_output = _cat_weighted1(C_output, subsection_embedding, input_ss)
    network = tf.matmul(SS_output, tf.transpose(section_embedding), name='section_weight')
    # network = tflearn.fully_connected(network, SEC_NUM, activation='softmax')  # not sure
    network = tflearn.softmax(network)
    network = tflearn.regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    return network


# add citation features
def create_citation2(in_dim, em_dim):
    # Building 'penNet'
    input_layer = tflearn.input_data(shape=[None, in_dim + G_NUM + SUBC_NUM + C_NUM + SUBS_NUM + Citation_Dim],
                                     name='input')
    input_p = input_layer[:, 0:in_dim]
    input_ss = input_layer[:, in_dim:in_dim + SUBS_NUM]
    input_c = input_layer[:, in_dim + SUBS_NUM:in_dim + SUBS_NUM + C_NUM]
    input_sc = input_layer[:, in_dim + SUBS_NUM + C_NUM:in_dim + SUBS_NUM + C_NUM + SUBC_NUM]
    input_g = input_layer[:, in_dim + SUBS_NUM + C_NUM + SUBC_NUM:in_dim + SUBS_NUM + C_NUM + SUBC_NUM + G_NUM]
    input_citation = input_layer[:, in_dim + SUBS_NUM + C_NUM + SUBC_NUM + G_NUM:]
    group_embedding = tf.Variable(tf.random_normal([G_NUM, em_dim]), name='group_embedding')
    subclass_embedding = tf.Variable(tf.random_normal([SUBC_NUM, em_dim]), name='subclass_embedding')
    class_embedding = tf.Variable(tf.random_normal([C_NUM, em_dim]), name='class_embedding')
    subsection_embedding = tf.Variable(tf.random_normal([SUBS_NUM, em_dim]), name='subsection_embedding')
    section_embedding = tf.Variable(tf.random_normal([SEC_NUM, em_dim]), name='section_embedding')
    # SG_output = _cat_weighted(X, subgroup_embedding, W_SG)
    textual_inf = tflearn.embedding(input_p, input_dim=110240, output_dim=128, name='word_embedding')
    textual_embedding = tflearn.lstm(textual_inf, em_dim, dropout=0.8, name='lstm_weight')
    network = tf.concat([textual_embedding, input_citation], 1)  # dimension=256=(d(t) + d(c))
    network = tflearn.fully_connected(network, em_dim, activation='softmax')
    # network = tflearn.bidirectional_rnn(network, tflearn.BasicLSTMCell(128), tflearn.BasicLSTMCell(128),
    #                                     name='bilstm_weight')
    # network = tflearn.dropout(network, 0.8)
    G_output = _cat_weighted2(network, group_embedding, input_g)
    SC_output = _cat_weighted2(G_output, subclass_embedding, input_sc)
    C_output = _cat_weighted2(SC_output, class_embedding, input_c)
    SS_output = _cat_weighted2(C_output, subsection_embedding, input_ss)
    network = tf.matmul(SS_output, tf.transpose(section_embedding), name='section_weight')
    # network = tflearn.fully_connected(network, SEC_NUM, activation='softmax')  # not sure
    network = tflearn.softmax(network)
    network = tflearn.regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    return network


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        # Store a validation accuracy threshold, which we can compare against
        # the current validation accuracy at, say, each epoch, each batch step, etc.
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        """
        This is the final method called in trainer.py in the epoch loop.
        We can stop training and leave without losing any information with a simple exception.
        """
        # print dir(training_state)
        print("Terminating training at the end of epoch", training_state.epoch)
        if training_state.val_acc >= self.val_acc_thresh and training_state.acc_value >= self.val_acc_thresh:
            raise StopIteration

    def on_train_end(self, training_state):
        """
        Furthermore, tflearn will then immediately call this method after we terminate training,
        (or when training ends regardless). This would be a good time to store any additional
        information that tflearn doesn't store already.
        """
        print("Successfully left training! Final model accuracy:", training_state.acc_value)


def train(network, x, y, val_x, val_y, modelfile):
    model = tflearn.DNN(network, tensorboard_dir="/data/users/lzh/bwu/model/penNet/tflearn_logs/",
                        checkpoint_path='/data/users/lzh/bwu/model/penNet/model.tfl.ckpt')
    if os.path.isfile(modelfile):
        model.load(modelfile)
    # model.fit(x, y, n_epoch=100, validation_set=(val_x, val_y), shuffle=True,
    #           show_metric=True, batch_size=64, snapshot_step=200,
    #           snapshot_epoch=False, run_id='penNet')  # epoch = 100
    model.fit(x, y, n_epoch=100, validation_set=(val_x, val_y), shuffle=True,
              show_metric=True, batch_size=64, run_id='penNet')  # epoch = 100
    # Save the model
    model.save(modelfile)
    print('Model storage is finished')


def train_predict1(network, x, y, val_x, val_y, modelfile, test_x, test_y):
    model = tflearn.DNN(network, tensorboard_dir="/data/users/lzh/bwu/model/penNet2/64_100_shuffle/tflearn_logs/",
                        checkpoint_path='/data/users/lzh/bwu/model/penNet2/64_100_shuffle/model.tfl.ckpt')
    if os.path.isfile(modelfile):
        model.load(modelfile)
    # model.fit(x, y, n_epoch=100, validation_set=(val_x, val_y), shuffle=True,
    #           show_metric=True, batch_size=64, snapshot_step=200,
    #           snapshot_epoch=False, run_id='penNet')  # epoch = 100
    model.fit(x, y, n_epoch=100, validation_set=(val_x, val_y), shuffle=True,
              show_metric=True, batch_size=64, run_id='penNet')  # epoch = 100
    # Save the model
    model.save(modelfile)
    print('Model storage is finished')
    test_predict = model.predict(test_x)
    test_predict_trans = [np.argmax(one_hot) for one_hot in test_predict]
    test_labels_trans = [np.argmax(one_hot) for one_hot in test_y]
    print_evaluation_scores(test_labels_trans, test_predict_trans)
    print('Model predict is finished')


# add early_stopping
def train_predict2(network, x, y, val_x, val_y, modelfile, test_x, test_y):
    model = tflearn.DNN(network, tensorboard_dir="/data/users/lzh/bwu/model/penNet3/citation/64_50_test_shuffle2/tflearn_logs/",
                        checkpoint_path='/data/users/lzh/bwu/model/penNet3/citation/64_50_test_shuffle2/model.tfl.ckpt')
    if os.path.isfile(modelfile):
        model.load(modelfile)
    early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.90)
    try:
        model.fit(x, y, validation_set=(val_x, val_y), n_epoch=100, shuffle=True,
                  snapshot_epoch=True,  # Snapshot (save & evaluate) model every epoch.
                  show_metric=True, batch_size=64, callbacks=early_stopping_cb, run_id='penNet')
    except StopIteration:
        print("OK, stop iterate!Good!")
    # model.fit(x, y, n_epoch=50, validation_set=(val_x, val_y), shuffle=True,
    #           show_metric=True, batch_size=64, callbacks=early_stopping_cb, run_id='penNet')  # epoch = 100
    # Save the model
    model.save(modelfile)
    print('Model storage is finished')
    test_predict = model.predict(test_x)
    test_predict_trans = [np.argmax(one_hot) for one_hot in test_predict]
    test_labels_trans = [np.argmax(one_hot) for one_hot in test_y]
    print_evaluation_scores(test_labels_trans, test_predict_trans)
    print('Model predict is finished')


def predict(network, modelfile, test_x, test_y):
    model = tflearn.DNN(network)
    model.load(modelfile)
    test_predict = model.predict(test_x)
    test_predict_trans = [np.argmax(one_hot) for one_hot in test_predict]
    test_labels_trans = [np.argmax(one_hot) for one_hot in test_y]
    print_evaluation_scores(test_labels_trans, test_predict_trans)
    print('Model predict is finished')


def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("accuracy:", accuracy)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)


def _cat_weighted1(patent, embedding, weight):
    p = tf.matmul(patent, tf.transpose(embedding))
    p1 = tf.multiply(weight, p)
    e = tf.matmul(p1, embedding)
    return e


def _cat_weighted2(patent, embedding, weight):
    p = tf.matmul(patent, tf.transpose(embedding))
    weight_soft = tf.nn.softmax(weight)  # 加一个softmax
    # weight_soft = tflearn.softmax(weight)  # 加一个softmax
    p1 = tf.multiply(weight_soft, p)
    e = tf.matmul(p1, embedding)
    return e


if __name__ == '__main__':
    feature_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/feature_citation/full2_100.npy'
    label_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/section/section.npy'
    model_path = '/data/users/lzh/bwu/model/penNet3/citation/64_100_shuffle/model.tfl'
    index_dim = 100
    embedding_dim = 128
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels \
        = load_data_shuffle2(feature_path, label_path)
    net = create_citation2(index_dim, embedding_dim)
    # net = create_penNet4(index_dim, embedding_dim)
    # train(net, train_features, train_labels, validation_features, validation_labels, model_path)
    # predict(net, model_path, test_features, test_labels)
    train_predict1(net, train_features, train_labels, validation_features, validation_labels,
                   model_path, test_features, test_labels)
