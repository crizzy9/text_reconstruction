
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers.core import Dense

from tensorflow.python.ops import lookup_ops

import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense
#import nltk
import pickle

# %matplotlib inline
import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab
from collections import Counter
import csv

# Seq2Seq Items
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense
import nltk



with open("drive/Colab Notebooks/Final Report NLP/vect_output_final/vocabulary.pickle", "rb") as fp:
  vocab = pickle.load(fp)

vocab_size = len(vocab)
vocab_size

with open("drive/Colab Notebooks/Final Report NLP/vect_output_final/embedding_matrix.pickle", "rb") as fp:
  embeddings = pickle.load(fp)

embeddings.shape

with open("drive/Colab Notebooks/Final Report NLP/vect_output_final/features_to_index.pickle", "rb") as fp:
  features_to_index = pickle.load(fp)

len(features_to_index)

with open("drive/Colab Notebooks/Final Report NLP/vect_output_final/features_to_embeddings.pickle", "rb") as fp:
  features_to_embeddings = pickle.load(fp)

vocab = list(features_to_index.keys())

embeddings[0].shape

vocab.append(("<s>", None, None))

vocab.append(("</s>", None, None))

embeddings.shape

sos_vec = np.random.rand(181,)

eos_vec = np.random.rand(181,)

embeddings1 = np.concatenate((embeddings, [sos_vec]))

embeddings1.shape

final_embeddings = np.concatenate((embeddings1, [eos_vec]))

final_embeddings.shape

final_vocab = []

for token in vocab:
  word = token[0]
  pos = token[1]
  ner = token[2]
  final_vocab.append("{}_{}_{}".format(word, pos, ner))

src_dictionary = {}

count = 0
for token in final_vocab:
  src_dictionary[token] = count
  count += 1

src_reverse_dictionary = dict(zip(src_dictionary.values(),src_dictionary.keys()))



with open("drive/Colab Notebooks/Final Report NLP/vect_output_final/indexed_corpus.pickle", "rb") as fp:
  indexed_corpus = pickle.load(fp)

indexed_corpus = indexed_corpus[0]

len(indexed_corpus)

main_dataset = []

for line in indexed_corpus:
  new_line = []
  for token_id in line:
    word = src_reverse_dictionary[token_id]
    new_line.append(word)
  main_dataset.append(' '.join(new_line))

with open("drive/Colab Notebooks/Final Report NLP/data3/text.txt", "w") as fp:
  for word in main_dataset:
    fp.write("%s\n" % word)

pickle.dump(main_dataset, open("drive/Colab Notebooks/Final Report NLP/data3/final_dataset.pickle", "wb" ))

pickle.dump(final_vocab, open("drive/Colab Notebooks/Final Report NLP/data3/vocab.pickle", "wb" ))

pickle.dump(src_dictionary, open("drive/Colab Notebooks/Final Report NLP/data3/src_dictionary.pickle", "wb" ))

pickle.dump(src_reverse_dictionary, open("drive/Colab Notebooks/Final Report NLP/data3/src_reverse_dictionary.pickle", "wb" ))

pickle.dump(final_embeddings, open("drive/Colab Notebooks/Final Report NLP/data3/embeddings.pickle", "wb" ))

with open("drive/Colab Notebooks/Final Report NLP/data3/new_vocab.txt", "w") as fp:
  for word in final_vocab:
    fp.write("%s\n" % word)


