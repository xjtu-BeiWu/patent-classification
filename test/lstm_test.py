import os

import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tflearn.data_utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# def word_count(file_name):
#     word_freq = collections.defaultdict(int)
#     with open(file_name) as f:
#         for l in f:
#             l = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", l).strip().split()
#             for w in l:
#                 word_freq[w] += 1
#     return word_freq
#
#
# def build_dict(file_name, min_word_freq=0):
#     word_freq = word_count(file_name)
#     word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items())
#     word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
#     words, _ = list(zip(*word_freq_sorted))
#     word_idx = dict(zip(words, xrange(len(words))))
#     word_idx['<unk>'] = len(words)
#     return word_idx


# def string2index(line, dict):
#     line = line.strip().split()
#     print(line)
#     for index in range(len(line)):
#         dict.get(line[index])
#
#
# def index_generation(inputPath):
#     text = open(inputPath)
#     dict = build_dict(inputPath)
#     output = open('', 'a+')
#     for line in text:
#         string2index(line, dict)

# if __name__ == '__main__':
#     with tf.Session() as sess:
#         # load the meta graph and weights
#         saver = tf.train.import_meta_graph('/data/users/lzh/jupyter/LZH/data/patent_landscaping/5.9m/'
#                                            'checkpoints/5.9m_abstracts.ckpt-1325000.meta')
#         saver.restore(sess, tf.train.latest_checkpoint('/data/users/lzh/jupyter/LZH/data/patent_landscaping/5.9m/'
#                                                        'checkpoints/'))
#
#         # get weights
#         graph = tf.get_default_graph()
#         # embedding_weights = graph.get_tensor_by_name('Variable:0')
#         # print(embedding_weights)
#         # print(sess.run(embedding_weights))
#         embedding_weights_2 = graph.get_tensor_by_name('Variable/read:0')
#         # print(embedding_weights_2)
#         # print(sess.run(embedding_weights_2))
#         nps = np.loadtxt('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/vocab_test_dict_6.txt')
#         data = np.frombuffer(nps).astype(np.int32)
#         input_data = data.reshape(2, 178)
#         word_embedding = tf.nn.embedding_lookup(embedding_weights_2, input_data)
#         print(word_embedding)
#         print(sess.run(word_embedding))
#         # embedding_weights_3 = graph.get_tensor_by_name('Variable_1/read:0')
#
#         # # print all the variables from pretrained-model
#         # # 首先，使用tensorflow自带的python打包库读取模型
#         # reader = pywrap_tensorflow.NewCheckpointReader('/data/users/lzh/jupyter/LZH/data/patent_landscaping/5.9m/'
#         #                                                'checkpoints/5.9m_abstracts.ckpt-1325000')
#         # # 然后，使reader变换成类似于dict形式的数据
#         # var_dict = reader.get_variable_to_shape_map()
#         # # 最后，循环打印输出
#         # for key in var_dict:
#         #     print("variable name: ", key)
#         #     print(reader.get_tensor(key))
#         # # print(embedding_weights)
#
# # # 定义一个未知变量input_ids用于存储索引
# #
# # input_ids = tf.placeholder(dtype=tf.int32, shape=[None])  # 定义一个已知变量embedding，是一个5*5的对角矩阵
# # # embedding = tf.Variable(np.identity(5, dtype=np.int32))
# #
# # # 或者随机一个矩阵
# # embedding = a = np.asarray([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3],
# #                                     [4.1, 4.2, 4.3]])  # 根据input_ids中的id，查找embedding中对应的元素
# # input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
# # sess = tf.InteractiveSession()
# # sess.run(tf.global_variables_initializer())  # print(embedding.eval())
# # print(sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))


if __name__ == '__main__':
    # with tf.Session() as sess:
    #     # load the meta graph and weights
    #     saver = tf.train.import_meta_graph('/data/users/lzh/jupyter/LZH/data/patent_landscaping/5.9m/'
    #                                        'checkpoints/5.9m_abstracts.ckpt-1325000.meta')
    #     saver.restore(sess, tf.train.latest_checkpoint('/data/users/lzh/jupyter/LZH/data/patent_landscaping/5.9m/'
    #                                                    'checkpoints/'))
    #
    #     # get weights
    #     graph = tf.get_default_graph()
    #     # embedding_weights = graph.get_tensor_by_name('Variable:0')
    #     # print(embedding_weights)
    #     # print(sess.run(embedding_weights))
    #     embedding_weights_2 = graph.get_tensor_by_name('Variable/read:0')
    #     # print(embedding_weights_2)
    #     # print(sess.run(embedding_weights_2))
    #     nps = np.loadtxt('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/vocab_test_dict_6.txt')
    #     data = np.frombuffer(nps).astype(np.int32)
    #     input_data = data.reshape(2, 178)
    #     word_embedding = tf.nn.embedding_lookup(embedding_weights_2, input_data)
    #     print(word_embedding)
    #     print(sess.run(word_embedding))
    # data = [[1], [2], [3], [4], [5], [6], [7], [7]]
    data = [[1, 2, 3, 4, 5, 6, 7, 7]]
    data = np.array(data)
    one_hots = to_categorical(data, nb_classes=8)
    print(one_hots)
    data = [np.argmax(one_hot) for one_hot in one_hots]
    print(data)
