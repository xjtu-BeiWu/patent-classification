# -*- coding: utf-8 -*-

import collections
import csv
import re
import time

import numpy as np
from sklearn.externals.joblib.numpy_pickle_utils import xrange
import string


def word_count(file_path):
    word_freq = collections.defaultdict(int)
    with open(file_path) as f:
        for l in f:
            l = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", l).strip().split()
            for w in l:
                word_freq[w] += 1
    return word_freq


def build_dict(file_path, min_word_freq=100):
    word_freq = word_count(file_path)
    word_freq = filter(lambda x: x[1] > min_word_freq, word_freq.items())
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, xrange(len(words))))
    word_idx['<unk>'] = len(words)
    return word_idx


def read_index_statistics():
    file_path = '/data/users/lzh/bwu/data/LOP/FirstStep/feature-04-lstm.txt'
    f = open(file_path)
    nums = []
    for line in f:
        arr = line.split(' ')
        # print(len(arr))
        nums.append(len(arr))
    # print(nums)
    print(np.where(nums == np.min(nums)))
    print(str(np.min(nums)))
    print(np.where(nums == np.max(nums)))
    print(str(np.max(nums)))
    f.close()


def load_dict(dict_path):
    # vocab = []
    file = open(dict_path, 'r', encoding='ISO-8859-1')
    # textline = file.read()
    csv_file = csv.reader(file)
    # encodingtype = detect(textline)['encoding']
    # print(encodingtype)
    # filecontent = textline.decode(encodingtype).encode('utf-8')
    # print(filecontent)
    # next(csv_file)
    header = next(csv_file)
    words = []
    for row in csv_file:
        words.append(row[1])
    # vocab = [[float(x) for x in row] for row in vocab]  # 将数据从string形式转换为float形式
    word_idx = dict(zip(words, xrange(len(words))))
    # dic = dict(vocab)
    # print(word_idx)
    file.close()
    return word_idx


def extract_data(filename, num_patents):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    print('Extracting', filename)
    data = np.loadtxt(filename)  # 从文件读取数据，存为numpy数组
    data = np.frombuffer(data).astype(np.int32)  # 改变数组元素变为int32类型
    data = data.reshape(num_patents, 100)  # 所有元素
    return data


def extract_labels(filename):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    labels = np.loadtxt(filename)
    labels = np.frombuffer(labels).astype(np.int32)
    # labels = labels.reshape(1, num_patents)  # 标签
    labels = labels.flatten()
    return labels


def trans(infile_path, outfile_path):
    input_file = open(infile_path)
    outfile = open(outfile_path, 'a+')
    for line in input_file:
        outfile.write(line.strip('\n'))
        outfile.write(' ')
    input_file.close()
    outfile.flush()
    outfile.close()


if __name__ == '__main__':
    # file_name = '/data/users/lzh/bwu/data/LOP/FirstStep-2/svm/data.json'
    file_name = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/thiPro/abstract.txt'
    # file_name = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/vocab_test.txt'
    vocab_path = '/data/users/lzh/jupyter/LZH/data/patent_landscaping/5.9m/vocab/vocab.csv'
    # outfile_name = '/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/data_100.txt'
    outfile_name = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/abstract_150.txt'
    # outfile_name = '/data/users/lzh/bwu/code/patentcls-multiclass/testfile/vocab_test_8.txt'
    word_dict = load_dict(vocab_path)
    # word_dict = build_dict(file_name, min_word_freq=100)
    file = open(file_name)
    outfile = open(outfile_name, 'a+')
    unk = 110239
    n = 0
    for line in file:
        line = line.lower().strip('\n')
        printable = string.digits + string.punctuation
        # print(printable)
        bytes_tabtrans = bytes.maketrans(b'ABCDEFGHIJKLMNOPQRSTUVWXYZ', b'abcdefghijklmnopqrstuvwxyz')
        # line = str(line).translate(line.translate(bytes_tabtrans, printable))
        line = bytes(line, encoding='utf-8').translate(bytes_tabtrans, bytes(printable, encoding='utf-8'))
        line = str(line)
        # print(line)
        line = line[2:len(line) - 1]
        # print(line)
        line = line.strip('').split()
        # line = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——！，。？、~@#￥%……&*（）]+\d+", " ", line).strip().split()
        for i in range(150):
            if i in range(len(line)):
                index = word_dict.get(line[i])
                # print(line[i].format())
                # print(index)
                if str(index) == 'None':
                    outfile.write(str(word_dict.get('UNK')))
                    # print('hello world')
                    outfile.write(' ')
                else:
                    outfile.write(str(index))
                    outfile.write(' ')
            else:
                outfile.write(str(unk))
                outfile.write(' ')
        n += 1
        # if n == 1033390:
        #     print(str(line))
        # if n == 1033391:
        #     print(str(line))
        outfile.write('\n')
    print('The number of instances is: ' + str(n))
    # # string1 = 'None'
    # # if string1 == 'None':
    # #     print('hello world!')
    # # print(word_dict)
    file.close()
    outfile.flush()
    outfile.close()
    # labels_filename = "/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/section/section-1.txt"
    # labels_lstmpath = "/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/label.txt"

    # start_time = time.time()
    # # Extract it into numpy arrays.
    # # data_all = extract_data(outfile_name, n)
    # # trans(labels_filename, labels_lstmpath)
    # labels_all = extract_labels(labels_filename)
    # # elapsed_time = time.time() - start_time
    # # print(elapsed_time, 's')
    # # start_time = time.time()
    # # features_file = np.save("/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/data_100.npy", data_all)
    # labels_file = np.save("/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/section/section.npy", labels_all)
    # elapsed_time = time.time() - start_time
    # print(elapsed_time, 's')

    data = np.load('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/section/section.npy').astype(np.int32)
    print(data)

    # read_index_statistics()
