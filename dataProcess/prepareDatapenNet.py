from bert_serving.client import BertClient
import numpy as np
import re
import time

np.set_printoptions(suppress=True)
bc = BertClient(ip='localhost', check_version=False, check_length=False)


def test():
    vec = bc.encode(['hello world', 'hello', 'world'])
    print(vec)

#
# def write_file(outsid):
#     output = open('', 'w')
#     output.write(outsid)
#     output.write("\n")


# 读取文本的前10行
def read():
    infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data-bert.txt')
    i = 0
    for line in infile:
        if i in range(10):
            print(line)
            print('---------------------------------')
            i += 1
    infile.close()


# 将文本转换成bert向量 300d
def read_and_write():
    infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/svm/data.txt')
    output_file = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data-bert.txt', 'a+')
    n = 0
    m = [0 for i in range(30)]
    j = 0
    for line in infile:
        if line != '\n':
            vec = bc.encode([line])
            s = str(vec).replace('\n', '').strip('[[').strip(']]')
            # s = str(vec).replace('\n', '').replace(' ', '').strip('[[').strip(']]')
            # s = re.compile(' ').sub('', str(vec).replace('\n', '').strip('[[').strip(']]'))
            output_file.write(s)
            output_file.write('\n')
            n = n + 1
            print(str(n))
        else:
            m[j] = n
            # print(line)
            # print('null-line is:' + str(n))
            j = j + 1
            # print('write....')
    print('The total number of patent instances is: ' + str(n))
    print(m)
    infile.close()
    output_file.flush()
    output_file.close()


# 去除各个数字之间的“,”
def reprocess():
    infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data-bert.txt')
    output = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data-bert-01.txt', 'a+')
    for line in infile:
        if line != '\n':
            s = line.replace(', ', ' ')
            output.write(s)
    infile.close()
    output.close()


# 将生成的txt文件转换成.npy文件
def extract_data(filename, num):
    """Extract the images into a 4D tensor [image index, y, x, channels]."""
    print('Extracting', filename)
    data = np.loadtxt(filename)  # 从文件读取数据，存为numpy数组
    data = np.frombuffer(data).astype(np.float32)  # 改变数组元素变为float32类型
    data = data.reshape(num, 768)  # 所有元素
    return data


if __name__ == "__main__":
    print('start......')
    # read_and_write()

    # reprocess()

    # Extract it into numpy arrays.
    start_time = time.time()
    infile_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data-bert.txt'
    outfile_path = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data-bert.npy'
    num_patents = 1033394
    features = extract_data(infile_path, num_patents)
    features_file = np.save(outfile_path, features)
    elapsed_time = time.time() - start_time
    print(elapsed_time, 's')

    # read()

    # test()

    print('end.......')
