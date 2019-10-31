import numpy as np
import pickle


def statistic():
    infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/group.txt')
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    n6 = 0
    n7 = 0
    n8 = 0
    n9 = 0
    n10 = 0
    for line in infile:
        line = line.strip('\n')
        line = line.split(' ')
        line = list(map(int, line))
        arr = np.array(line)
        if (np.sum(arr == 1)) == 1:
            n1 += 1
        if (np.sum(arr == 1)) == 2:
            n2 += 1
        if (np.sum(arr == 1)) == 3:
            n3 += 1
        if (np.sum(arr == 1)) == 4:
            n4 += 1
        if (np.sum(arr == 1)) == 5:
            n5 += 1
        if (np.sum(arr == 1)) == 6:
            n6 += 1
        if (np.sum(arr == 1)) == 7:
            n7 += 1
        if (np.sum(arr == 1)) == 8:
            n8 += 1
        if (np.sum(arr == 1)) == 9:
            n9 += 1
        if (np.sum(arr == 1)) == 10:
            n10 += 1
    print('The count of 1 multi-class instances is:' + str(n1))
    print('The count of 2 multi-class instances is:' + str(n2))
    print('The count of 3 multi-class instances is:' + str(n3))
    print('The count of 4 multi-class instances is:' + str(n4))
    print('The count of 5 multi-class instances is:' + str(n5))
    print('The count of 6 multi-class instances is:' + str(n6))
    print('The count of 7 multi-class instances is:' + str(n7))
    print('The count of 8 multi-class instances is:' + str(n8))
    print('The count of 9 multi-class instances is:' + str(n9))
    print('The count of 10 multi-class instances is:' + str(n10))
    infile.close()


def readtxt():
    infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/lstm/data_index.txt')
    index = 0
    for line in infile:
        if index in range(10):
            print(line)
            print('---------------------------------')
            index += 1
    infile.close()


def read_pkl():
    infile = open('/data/users/lzh/jupyter/LZH/data/patent_landscaping/5.9m/train_words.pkl', 'rb')
    in_data = pickle.load(infile)
    index = 0
    for line in in_data:
        if index in range(10):
            print(line)
            index += 1
    # print(in_data)
    infile.close()


def readasnp():
    nps = np.loadtxt('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/vocab_test_dict_6.txt')
    data = np.frombuffer(nps).astype(np.int)
    data = data.reshape(2, 178)
    print(data)
    # num = 0
    # for n in data:
        # num += 1
        # print(str(n))
        # print(len(n))
    # print(str(num))


if __name__ == '__main__':
    statistic()
    # readtxt()
    # read_pkl()
    # readasnp()
