import os
import time

import numpy as np
import tflearn
from sklearn.metrics import accuracy_score, f1_score
from tflearn.data_utils import to_categorical, pad_sequences
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# CLASS_NUMBER = 8
EPOCH_SIZE = 200
BATCH_SIZE = 100

TRAIN_SIZE = 990000
VALIDATION_SIZE = 10000
PRE_TRAINED_MODEL_PATH = "/data/users/lzh/jupyter/LZH/data/patent_landscaping/5.9m/checkpoints/"


def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("accuracy:", accuracy)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)


def load_pretrained_model(meta_path, checkpoint_path):
    with tf.Session() as sess:
        # load the meta graph and weights
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        # get weights
        graph = tf.get_default_graph()
        embedding_weights = graph.get_tensor_by_name('Variable/read:0')
        # print(embedding_weights.shape)
    return embedding_weights


# Data loading and preprocessing
start_time = time.time()
features = np.load("/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/data_150.npy").astype(dtype=np.int32)
labels = np.load("/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/label.npy").astype(dtype=np.int32)
elapsed_time = time.time() - start_time
print('read data time: ', elapsed_time, 's')

# Generate a validation set.
labels = to_categorical(labels, nb_classes=8)
print('Labels transformation is finished!')
train_data = features[:TRAIN_SIZE]
train_labels = labels[:TRAIN_SIZE]
validation_data = features[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
validation_labels = labels[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE]
test_data = features[TRAIN_SIZE + VALIDATION_SIZE:]
test_labels = labels[TRAIN_SIZE + VALIDATION_SIZE:]
print('Dataset setting is finished!')

# train_X = pad_sequences(train_data, maxlen=150, value=110239)
# validation_X = pad_sequences(validation_data, maxlen=150, value=110239)
# test_X = pad_sequences(test_data, maxlen=150, value=110239)
# train_Y = to_categorical(train_labels, nb_classes=8)
# validation_Y = to_categorical(validation_labels, nb_classes=8)
# test_Y = to_categorical(test_labels, nb_classes=8)

print('-----------------start training-----------------')
start_time = time.time()
# Building deep neural network
net = tflearn.input_data(shape=[None, 150], dtype=tf.int32)
embeddings = load_pretrained_model(PRE_TRAINED_MODEL_PATH + "5.9m_abstracts.ckpt-1325000.meta", PRE_TRAINED_MODEL_PATH)
# print(embeddings)
# net = tflearn.embedding(net, weights_init=embeddings, input_dim=110240, output_dim=300, trainable=False)
# net = tflearn.embedding(net, input_dim=110240, output_dim=300, trainable=False, restore=True)
# net = tflearn.embedding(net, weights_init=[embedding], trainable=False)
net = tf.nn.embedding_lookup(embeddings, net)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 8, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir="/data/users/lzh/bwu/model/lstm/150/tflearn_logs/",
                    checkpoint_path='/data/users/lzh/bwu/model/lstm/150/checkpoints/')
model.fit(train_data, train_labels, validation_set=(validation_data, validation_labels), show_metric=True,
          batch_size=32)
elapsed_time = time.time() - start_time
print('Training time: ', elapsed_time, 's')
model.save('/data/users/lzh/bwu/model/lstm/150/model.tfl')
elapsed_time = time.time() - start_time
print('Model storage is finished')

test_predict = model.predict(test_data)
print('Model test is finished')
test_predict_trans = [np.argmax(one_hot) for one_hot in test_predict]
test_labels_trans = [np.argmax(one_hot) for one_hot in test_labels]
print_evaluation_scores(test_labels_trans, test_predict_trans)
# accuracy = accuracy_score(test_labels, test_predict)
# print('Accuracy: ' + str(accuracy))
