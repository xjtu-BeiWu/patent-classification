import langid

# # 生成fasttext分类要求的文本格式，以‘__label__’为前缀，后接类别
# infile_label = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/section/section-1.txt')
# infile_data = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/thiPro/abstract.txt')
# outfile = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext2/train-fasttext.txt', 'a+')
# n = 0
# for line1, line2 in zip(infile_label, infile_data):
#     # print(line1)
#     # print(line2)
#     outfile.write('__label__')
#     outfile.write(str(line1).strip('\n'))
#     outfile.write(' ')
#     outfile.write(line2)
#     # outfile.write('\n')
#     n += 1
# print('The number of instances is: ' + str(n))
# infile_data.close()
# infile_label.close()
# outfile.flush()
# outfile.close()

# # 去除重复数据，以及非英语数据
# input_file = open('/data/users/lzh/bwu/data/LOP/FirstStep-3/fasttext/train-fasttext.txt')
# out_file = open('/data/users/lzh/bwu/data/LOP/FirstStep-3/fasttext/train-fasttext-1.txt', 'a+')
# lines_seen = set()
# n = 0
# for line in input_file:
#     line = line.strip()
#     lineTuple = langid.classify(line)
#     if lineTuple[0] == 'en':  # 判断是否为英语数据
#         if line not in lines_seen:  # 判断是否为非重复数据
#             out_file.write(line)
#             lines_seen.add(line)
#             out_file.write('\n')
#             n += 1
# print('***************************************')
# print('The number of instances is: ' + str(n))
# print('***************************************')
# input_file.close()
# out_file.flush()
# out_file.close()

# 将数据集分成训练数据集和测试数据集
infile = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext2/train-fasttext.txt')
outfile_train = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext2/train_val.txt', 'a+')
outfile_test = open('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext2/test.txt', 'a+')
train_num = 1370000
index = 0
for line in infile:
    if index < train_num:
        outfile_train.write(line)
        index += 1
    else:
        outfile_test.write(line)
        index += 1
infile.close()
outfile_test.flush()
outfile_test.close()
outfile_train.flush()
outfile_train.close()
