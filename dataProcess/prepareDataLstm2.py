from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# the model is loaded. It can be used to perform all of the tasks mentioned above.

dog = model['dog']
print(model.similarity('8', 'eight'))

# from gensim.test.utils import common_texts, get_tmpfile
# from gensim.models import Word2Vec
#
# path = get_tmpfile("word2vec.model")
# model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")
# model = Word2Vec.load("word2vec.model")
#
# vector1 = model.wv['nine']
# vector2 = model.wv['eight']
