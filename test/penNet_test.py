import re
import os
import tflearn
import tensorflow as tf

# test = 'I  love python'
# str_info = re.compile(' ')
# test = str_info.sub('', test)
# print(test)

# weight = [[[0], [0.1], [0.2], [0.7]],
#           [[0], [0], [0.2], [0.8]],
#           [[0], [0], [1.0], [0]],
#           [[0.6], [0], [0.4], [0]],
#           [[0.5], [0.5], [0], [0]]]
# # weight = [[0, 0.1, 0.2, 0.7], [0, 0, 0.2, 0.8], [0, 0, 1.0, 0], [0.6, 0, 0.4, 0], [0.5, 0.5, 0, 0]]

# t2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]]
# t2 = tf.to_float(t2)
# # a = tflearn.merge([t1, weight, t2], 'elemwise_mul', axis=1, name="Merge")
# # b = tflearn.merge([weight, t1], 'prod', axis=1, name="Merge")
# # print(a)
# b = tf.multiply(weight, t1)
# c = tf.multiply(t1, t2)
# # b = tf.matmul(weight, t1)
# print(b)
# sess = tf.Session()
# print(sess.run(b))

# t1 = [1, 2, 3]
# t2 = [1, 2, 3]
# t1_t2 = tf.multiply(t1, t2)
# t = tf.reduce_sum()
# sess = tf.Session()
# print(sess.run(t1_t2))
# patent = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6]]
# embedding = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]]
# patent = tf.to_float(patent)
# embedding = tf.to_float(embedding)
#
# weight1 = [[0, 0, 1, 1],
#            [1, 0, 0, 0],
#            [0, 1, 0, 0],
#            [0, 0, 1, 0],
#            [1, 1, 0, 0]]
# weight = tf.to_float(weight1)
#
# p = tf.matmul(patent, tf.transpose(embedding))
# p1 = tf.multiply(weight, p)
# p2 = tf.matmul(p1, embedding)
# # embedding = tf.multiply(weight, t1)
# sess = tf.Session()
# input = weight[:, 0:2]
# input2 = weight[:, 2:]
# # print(patent.get_shape())
# print(sess.run(input))
# print(sess.run(input2))

s = tf.constant([[1,  2, 3, 4, 5, 10], [1, 1, 2, 3, 4, 5, 10]], dtype=tf.float32)
sm = tf.nn.softmax(s)

with tf.Session()as sess:
    print(sess.run(sm))
