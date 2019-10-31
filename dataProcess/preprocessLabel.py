def patent_type_trans(s):
    it = {'1 0 0 0 0 0 0 0': 0,
          '0 1 0 0 0 0 0 0': 1,
          '0 0 1 0 0 0 0 0': 2,
          '0 0 0 1 0 0 0 0': 3,
          '0 0 0 0 1 0 0 0': 4,
          '0 0 0 0 0 1 0 0': 5,
          '0 0 0 0 0 0 1 0': 6,
          '0 0 0 0 0 0 0 1': 7}
    return it[s]


def patent_type_trans2(s):
    it = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    return it[s]


inputPath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/forPro/section.txt'
outPath = '/data/users/lzh/bwu/data/LOP/FirstStep-2/penNet/data/section/section-1.txt'
labels = open(inputPath)
outfile = open(outPath, 'a+')
label = []
for line in labels:
    line = line.strip('\n')
    # print(patent_type_trans(line))
    label.append(patent_type_trans(line))
for index in range(len(label)):
    outfile.write(str(label[index]))
    # outfile.write(' ')
    outfile.write('\n')
    # print(str(label[index]))
# outfile.write('\n')
labels.close()
outfile.flush()
outfile.close()
