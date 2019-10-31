# 将处理过的数据转换成两个文件，一个是label，一个是文本数据
infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext/train-fasttext.txt')
outfile_data = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/svm/data.txt', 'a+')
outfile_label = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/svm/label.txt', 'a+')
# infile = open('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/test')
# outfile_data = open('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/data.txt', 'a+')
# outfile_label = open('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/label.txt', 'a+')
for line in infile:
    outfile_label.write(line[9])
    outfile_label.write('\n')
    outfile_data.write(line[11:])
infile.close()
outfile_data.flush()
outfile_data.close()
outfile_label.flush()
outfile_label.close()


# string = '__label__1 hello world and welcome to the internet'
# print(string[9])
# print(string[11:])
