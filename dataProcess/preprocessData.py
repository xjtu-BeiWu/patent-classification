import numpy as np


def statistic():
    infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-3/label/section-02.txt')
    outfile = open('/data/users/lzh/bwu/data/LOP/FirstStep-3/label/section-02-01.txt', 'a+')
    outfile2 = open('/data/users/lzh/bwu/data/LOP/FirstStep-3/label/single.txt', 'a+')
    n = 0
    num = []
    for line in infile:
        line = line.strip('\n')
        line = line.split(' ')
        line = list(map(int, line))
        arr = np.array(line)
        n += 1
        if (np.sum(arr == 1)) == 1:
            outfile.write(str(arr).strip('[').strip(']'))
            outfile.write('\n')
            outfile2.write(str(n))
            outfile2.write('\n')
            num.append(n)
    infile.close()
    outfile.flush()
    outfile.close()
    outfile2.flush()
    outfile2.close()
    return num


def write():
    num = np.array(statistic())
    print(num)
    infile_data = open('/data/users/lzh/bwu/data/LOP/Origin/LOP-All-Abstract-03.txt')
    outfile_data = open('/data/users/lzh/bwu/data/LOP/FirstStep-3/feature.txt', 'a+')
    n = 0
    for line in infile_data:
        line = line.strip('\n')
        n += 1
        if (num == n).any():
            outfile_data.write(line)
            outfile_data.write('\n')
    infile_data.close()
    outfile_data.flush()
    outfile_data.close()


def read():
    infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-3/section-02-01.txt')
    n = 0
    for line in infile:
        print(line)
        if len(line):
            n += 1
        # print('---------------------------------')
    print('The line number of the file is: ' + str(n))
    print(n)
    infile.close()


def load():
    # infile = np.loadtxt('/data/users/lzh/bwu/data/LOP/FirstStep/section-02-01.txt', dtype=float, delimiter=' ')
    infile = np.loadtxt('/data/users/lzh/bwu/data/LOP/FirstStep/section-02-02.txt', dtype=float)
    print(infile)
    print(infile.dtype)
    print(infile.shape)


if __name__ == '__main__':
    # statistic()
    # read()
    write()
    # load()
