# -*- coding: utf-8 -*-
# @Time : 7/16/19 4:14 PM
# @Author : Bei Wu
# @Site : 
# @File : hash_test.py
# @Software: PyCharm
from keras_preprocessing.text import hashing_trick
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import FeatureHasher
import numpy as np
from keras.preprocessing import text
import json

# vectorizer = CountVectorizer()
# vectorizer2 = HashingVectorizer(n_features=6, norm=None)

# # vectorizer3 = HashingVectorizer(n_features=2**4)
#
# corpus = ["I come to China to travel",
#           "This is a car polupar in China",
#           "I love tea and Apple ",
#           "The work is to write some papers in science"]
#
# print(vectorizer.fit_transform(corpus))
# print(vectorizer.fit_transform(corpus).toarray())
# print(vectorizer.get_feature_names())
# X0 = vectorizer.fit_transform(corpus)
# print(X0.shape)
# print(vectorizer2.fit_transform(corpus))
# X1 = vectorizer2.fit_transform(corpus)
# print(X1.toarray())
# print(vectorizer2.transform(corpus).todense())
# # print(X1.get_feautre_names())
# print(X1.shape)
# print()

# print(vectorizer3.fit_transform(corpus))
# X2 = vectorizer3.fit_transform(corpus)
# print(X2.shape)
corpus2 = [[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]]
# print(vectorizer2.fit_transform(corpus2))

# string = "http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27344, " \
#          "http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/32C0ACCC76C369059B7ECA42FE32DE43, "
# strs = string.strip(', ').replace('\n', '').split(', ')
# for i in range(len(strs)):
#     print(strs[i])

# corpus2 = ['http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27344,'
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27342,'
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27341,'
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27300',
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27349,'
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27344,'
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27346,'
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27345',
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27344,'
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27343',
#            'http://data.epo.org/linked-data/data/publication/EP/0989583/A1/-/citation/130EB59A4CE65A73FA0CDD2EE9A27341']

corpus3 = ['23,1,3,4',
           '1,6',
           '3,0,1',
           '',
           '222256,6,2098754',
           '6,7']
corpus4 = ['23 1 3 4',
           '1 6',
           '3 0 1',
           '',
           '222256 6 2098754',
           '6 7']
#
# vectorizer2 = HashingVectorizer(n_features=6, lowercase=False)
# X = vectorizer2.fit_transform(corpus3)
# print(X.toarray())
#
# vectorizer4 = CountVectorizer(lowercase=False)
# X2 = vectorizer4.fit_transform(corpus3)
# print(X2.toarray())
# print(vectorizer4.get_feature_names())

# X_train = np.array([[1., 1.], [2., 3.], [4., 0.]])
# X_train_crs = csr_matrix(X_train)
# print(X_train_crs)
# print(X_train_crs.toarray())
# n_features = X_train.shape[1]
# print(n_features)
# n_desired_features = 3
# buckets = np.random.random_integers(0, n_desired_features - 1, size=n_features)
# X_new = np.zeros((X_train.shape[0], n_desired_features), dtype=X_train.dtype)
# for i in range(n_features):
#     X_new[:, buckets[i]] += X_train[:, i]
# print(X_new)
# M = coo_matrix((np.repeat(1, n_features), (range(n_features), buckets)),
#                shape=(n_features, n_desired_features))
# X_new = X_train.dot(M)
# print(X_new)


# def feature_hashing(features, m):
#     """
# 	Args:
# 		features: 输入特征序列，可以是数字，字符串(本质还是数字)
# 		m: 输出的特征维度，通常是2**26(vw),2**20(scikit-learn)
# 	Returns:
# 		m维的（稀疏）向量
# 	"""
#     # 这里的x一般是稀疏表示的（如：scipy.sparse.csr_matrix），这里是为了方便说明
#     x = np.zeros(m)
#     for feature in features:
#         idx = hash_func_1(feature) % m
#         sign = hash_func_2(feature) % 2
#         if sign == 1:
#             x[idx] += 1
#         else:
#             x[idx] -= 1
#     return x

# h = FeatureHasher(n_features=2, input_type='string')
# D = [{'dog': 1, 'cat': 2, 'elephant': 4}, {'dog': 2, 'run': 5}]
# f = h.transform(D)
# print(f.toarray())
# import sklearn.preprocessing.OneHotEncoder

refs_vocab_size = 10


def tokenize_to_onehot_matrix(text_series, vocab_size, keras_tokenizer=None):
    '''
    '''
    if keras_tokenizer is None:
        print('No Keras tokenizer supplied so using vocab size ({}) and series to build new one'.format(vocab_size))

        keras_tokenizer = text.Tokenizer(
            num_words=vocab_size,
            split=' ',
            # filter should be same as default, minus the '-'
            filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n',
            lower=False)
        keras_tokenizer.fit_on_texts(text_series)
        keras_tokenizer.index_word = {idx: word for word, idx in keras_tokenizer.word_index.items()}

    # text_one_hot = keras_tokenizer.texts_to_matrix(text_series, mode='freq')
    text_one_hot = keras_tokenizer.texts_to_matrix(text_series)

    return keras_tokenizer, text_one_hot


def readTxt(filepath):
    file = open(filepath)
    li = []
    for line in file:
        li.append(line.strip('\n').strip(' '))
    file.close()
    return li
    # with open(filepath,'r') as f:
    #     l = f.readlines()


if __name__ == '__main__':
    # filepath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/citation/' \
    #            'citesPatentPublication/citesPatentPublication_feature.txt'
    word_counts_path = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/1.json'
    word_docs_path = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/2.json'
    word_index_path = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/3.json'
    hash_feature_path = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/4.txt'
    filepath_test = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/test_mon.txt'
    li = readTxt(filepath_test)
    refs_tokenizer, refs_one_hot = tokenize_to_onehot_matrix(li, refs_vocab_size)

    wc = json.dumps(refs_tokenizer.word_counts)
    word_counts_file = open(word_counts_path, 'w')
    word_counts_file.write(wc)

    wd = json.dumps(refs_tokenizer.word_docs)
    word_docs_file = open(word_docs_path, 'w')
    word_docs_file.write(wd)

    wi = json.dumps(refs_tokenizer.word_index)
    word_index_file = open(word_index_path, 'w')
    word_index_file.write(wi)

    np.savetxt(hash_feature_path, refs_one_hot)

    # refs_tokenizer, refs_one_hot = hashing_trick(corpus2, refs_vocab_size)
    # print(refs_one_hot)
    # print(refs_one_hot)
    # print(type(corpus4))
    word_counts_file.flush()
    word_counts_file.close()
    word_docs_file.flush()
    word_docs_file.close()
    word_index_file.flush()
    word_index_file.close()

