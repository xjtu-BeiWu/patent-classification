from gensim.models import Word2Vec

model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#the model is loaded. It can be used to perform all of the tasks mentioned above.

dog = model['dog']
print(model.similarity('woman', 'man'))