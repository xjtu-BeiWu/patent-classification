import logging
from fasttext import train_supervised, load_model

# basedir = '/Users/derry/Desktop/Data/'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

classifier = train_supervised('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext2/train_val.txt', label='__label__')
classifier.save_model('/data/users/lzh/bwu/model/fasttext2/ft.model')
result = classifier.test('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext2/test.txt')

print(classifier.test_label('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext/test.txt', 1))
# classifier = load_model('/data/users/lzh/bwu/model/fasttext/f100-2.model')
# classifier.predict('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext/test.txt')
# print(classifier.get_output_matrix('/data/users/lzh/bwu/data/LOP/FirstStep-2/fasttext/test.txt'))
# print(result)
