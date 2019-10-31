# coding=utf-8
import langid  # 引入langid模块
import numpy as np

#
# s1 = '你好'
# s2 = '__label__2 world'
# s2_1 = '__label__2 hello world'
# s3 = 'Flüssigkeiten zum Nassbehandeln von Wäschestücken werden vielfach mit Dampf aufgeheizt. Dazu wird der Dampf mit ' \
#      'hoher Geschwindigkeit durch eine Düse (30) der aufzuheizenden Flüssigkeit direkt zugeführt. Aufgrund der hohen ' \
#      'Geschwindigkeit, mit der der Dampf in die aufgeheizte Flüssigkeit einströmt, entstehen starke Geräusche sowie ' \
#      'Schwingungen und Vibrationen. Um mindestens die Geräusche zu reduzieren, ist es bereits bekannt, zusätzlich ' \
#      'Druckluft zuzuführen. Das verschlechtert den Wärmeübergang. Die Erfindung sieht es vor, in die Düse (30) eine ' \
#      'kleine Menge der aufzuheizenden Flüssigkeit einzusaugen und dadurch in der Düse (30) ein Kondensat-Dampfgemisch ' \
#      'zu bilden. Alternativ oder zusätzlich kann hinter der Düse (30) ein Strömungsteiler vorgesehen sein, ' \
#      'der die Strömungsgeschwindigkeit des Dampfs bzw. Dampf-Kondensatgemisches erhöht. Hierdurch und/oder durch die ' \
#      'Bildung eines Dampf-Kondensatgemisches in der Düse (30) werden die Geräuschentwicklung beim Einleiten des ' \
#      'Dampfs in die aufzuheizende Flüssigkeit sowie Schwingungen und Vibrationen ohne die Zufuhr von Druckluft ' \
#      'verringert '
#
# i = langid.classify(s1)
# j = langid.classify(s2)
# j_1 = langid.classify(s2_1)
# print(j)
# print(j_1)
# A = np.zeros(128)
# print(str(A).strip().replace('[', '').replace(']', '').replace('\n', ''))
# B = np.ones(128)
# print(str(B).strip().replace('[', '').replace(']', '').replace('\n', ''))
#
# np.random.seed(200)
# np.random.shuffle(train_data)
# np.random.seed(200)
# np.random.shuffle(train_label)
#
# np.random.seed(200)
# np.random.shuffle(test_data)
# np.random.seed(200)
# np.random.shuffle(test_label)

# m = langid.classify(s3)

# print(m, m[0], type(m))
#
# input_file = open('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/test')
# out_file = open('/data/users/lzh/bwu/code/patentcls-multiclass/testfile/test3.txt', 'a+')
# lines_seen = set()
# for line in input_file:
#     line = line.strip()
#     lineTuple = langid.classify(line)
#     if lineTuple[0] == 'en':
#         if line not in lines_seen:
#             out_file.write(line)
#             lines_seen.add(line)
#             out_file.write('\n')
# input_file.close()
# out_file.flush()
# out_file.close()


W_train = np.array([[1.1, 2.0, 3.0293485],
           [4, 5, 6.1029405],
           [7, 8, 9.1242231234],
           [10, 11, 12],
           [13, 14, 15]]).astype(dtype=np.float32)
R_train = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0]]).astype(dtype=np.int32)
data_train = [(w, r) for w, r in zip(W_train, R_train)]
np.random.shuffle(data_train)
print(data_train)
W_train = np.array([w for w, r in data_train]).astype(dtype=np.float32)
R_train = [r for w, r in data_train]
print(W_train)
print(R_train)

