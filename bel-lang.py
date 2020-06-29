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

trainBelDatasetLines = open(os.path.join('./resources/', 'train-bel.txt'), encoding="utf-8").read().split("\n")
# trainBelDatasetLines = open(os.path.join('./resources/', 'small.txt'), encoding="utf-8").read().split("\n")
belSymbols = frozenset(['а','б','в','г','д','е','ё','ж','з','і','й','к','л','м','н','о','п','р','с','т','у','ў','ф','х','ц','ч','ш','ы','ь','э','ю', 'я', '’'])

# tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n—…–‑0123456789«»abcdefghijklmnopqrstuvwxyz„‟’“¬”°ფეářэ́კო', lower=True,
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n—…–‑0123456789«»abcdefghijklmnopqrstuvwxyzა©§‹əęὐί¶ρ\x98µ®რí³""υż·იºř±´ŭἀ”კι\uf0fc‰ე意›\u200b你јý˚\u202cé‡αß¬ς́á\ufeffč¦џѓ←έω\uf020†„€ʌњђ\u200e‚ѕ∙љʊფ²πოҷ°“ëšʼļ\u202a的̓ν−•™εїēüόʃє№ć�🌿η\uf008მˈ同\x03ћοќÿ我łґ於óώ̆ž對‘ąś¤ѣ;×\'', lower=True,
    split=' ', char_level=False
)
# sequences = tokenizer.texts_to_sequences(trainBelDataset)
# print(sequences)
tokenizer.word_index
tokenizer.fit_on_texts(trainBelDatasetLines)
sequences = tokenizer.texts_to_sequences(trainBelDatasetLines)

wordsBeginnings = set()
wordsEndings = set()
for word in tokenizer.word_index:
    if len(word) > 2:
      wordsBeginnings.add(word[ 0 : 3 ])
      wordsEndings.add(word[-3:])

blackListChars = set()

with open(os.path.join('./resources/', 'endings-3.txt'), 'w', encoding="utf-8") as f:
    for word in wordsEndings:
        f.write("%s\n" % word)

with open(os.path.join('./resources/', 'beginnings-3.txt'), 'w', encoding="utf-8") as f:
    for word in wordsBeginnings:
        f.write("%s\n" % word)
        # for char in word:
        #     if not char in belSymbols:
        #       blackListChars.add(char)

print(blackListChars)

# wordsBeginnings = {x for x in wordsBeginnings if len(x) > 1}
# wordsEndings = {x for x in wordsEndings if len(x) > 1}


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