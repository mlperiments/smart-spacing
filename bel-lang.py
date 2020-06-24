import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer

# trainBelDataset = tf.data.TextLineDataset(os.path.join('./resources/', 'train-bel.txt'))
# trainBelDataset.map(lambda string: string)
# trainBelDataset = np.genfromtxt(os.path.join('./resources/', 'train-bel.txt'), dtype=None, usecols=(1, 3), encoding='UTF-8')

trainBelDatasetLines = open(os.path.join('./resources/', 'small.txt'), encoding="utf-8").read().split("\n")

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n—', lower=True,
    split=' ', char_level=False
)
# sequences = tokenizer.texts_to_sequences(trainBelDataset)
# print(sequences)

tokenizer.fit_on_texts(trainBelDatasetLines)
sequences = tokenizer.texts_to_sequences(trainBelDatasetLines)
word_index = tokenizer.word_index
print(word_index)

# Load Inutition Engineering pretrained model
# Models names: 'eng_50', 'eng_100', 'eng_150' 'eng_200', 'eng_300'
# c2v_model = chars2vec.load_model('eng_50')

# words = [
#   'NaturalLanguage',
#   'NaturalLanguageAndSomethingElse',
#   'NaturalLangu',

#   'NaturalUnderstanding',
#   'NaturalUnderstandingAnd',
#   'NaturalUnderstandingAndBla',
#   'NaturalUnderstandingAndBlaBla',
#   'NaturalUnderstandingAndSomethingElse',
# ]

# # Create word embeddings
# word_embeddings = c2v_model.vectorize_words(words)

# # Project embeddings on plane using the PCA
# projection_2d = sklearn.decomposition.PCA(n_components=2).fit_transform(word_embeddings)

# # Draw words on plane
# f = plt.figure(figsize=(8, 6))

# for j in range(len(projection_2d)):
#     plt.scatter(projection_2d[j, 0], projection_2d[j, 1],
#                 marker=('$' + words[j] + '$'),
#                 s=500 * len(words[j]), label=j,
#                 facecolors='green' if words[j] 
#                            in ['Natural', 'Language', 'Understanding'] else 'black')
# plt.show()