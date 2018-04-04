# Library Imports
import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab
from collections import Counter
import csv
from os import listdir
from os.path import isfile, join
from src.utils import store_pickle, load_pickle, abspath
from nltk import sent_tokenize
from collections import Counter

# Model Imports (Seq2Seq)
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense


class Modes():
    min_unk_frequency = 5
    lowercase = 0


class Parser:

    START_TOKEN = '<s>'
    END_TOKEN = '</s>'
    UNKNOWN_TOKEN = '<unk>'

    def __init__(self):
        print('Enter init...')

        if not os.path.isfile(abspath('out', 'parsed_files_fetch_data.pickle')):
            self.fetch_data()
        parsed_files = load_pickle(abspath('out', 'parsed_files_fetch_data.pickle'))

        if not os.path.isfile(abspath('out', 'parsed_files_clean_data.pickle')):
            self.clean_data(parsed_files)
        parsed_files = load_pickle(abspath('out', 'parsed_files_clean_data.pickle'))

        if not os.path.isfile(abspath('out', 'vocab_list.pickle')):
            self.parse_data(parsed_files)
        vocab_list = load_pickle(abspath('out', 'vocab_list.pickle'))
        vocab_freq_dict = load_pickle(abspath('out', 'vocab_freq_dict.pickle'))
        index_word_dict = load_pickle(abspath('out', 'index_word_dict.pickle'))
        word_index_dict = load_pickle(abspath('out', 'word_index_dict.pickle'))

    def fetch_data(self):
        print('Retrieving data from Gutenberg corpus...')
        train_directory = abspath('dataset', 'Gutenberg', 'txt')
        train_file_locations = [os.path.join(train_directory, f) for f in listdir(train_directory) if
                               isfile(join(train_directory, f)) and f != '.fuse_hidden0000465600000001']
        parsed_files = ''
        i = 0
        for file in train_file_locations:
            if i == 10:
                break
            i += 1
            with open(file, 'r', errors='ignore') as f:
                parsed_files += f.read()

        store_pickle(parsed_files, abspath('out', 'parsed_files_fetch_data.pickle'))

    def clean_data(self, parsed_files):
        print('Cleaning data...')

        # Convert to lower case if Modes.lowercase is 1
        if Modes.lowercase:
            parsed_files = parsed_files.lower()

        # Replace all spaces with a single space
        parsed_files = ' '.join(parsed_files.split())

        # Append start and end tags to every sentence
        parsed_files = sent_tokenize(parsed_files)
        parsed_files = ' '.join([self.START_TOKEN + ' ' + parsed_files[i] + ' ' + self.END_TOKEN
                                 for i in range(len(parsed_files))])

        # Replace all tokens with a count of 10 or less with the out-of-vocabulary symbol UNK.
        parsed_files = ' ' + parsed_files + ' '  # Append spaces for easy replacement
        for key, value in Counter(parsed_files.split(' ')).items():
            if value < Modes.min_unk_frequency:
                parsed_files = parsed_files.replace(' ' + key + ' ', ' ' + self.UNKNOWN_TOKEN + ' ')
                parsed_files = parsed_files.replace(' ' + key + ' ', ' ' + self.UNKNOWN_TOKEN + ' ')  # For multiple consecutive occurrences

        # Replace all spaces with a single space
        parsed_files = ' '.join(parsed_files.split())

        # Replace all blank sentences with a single space
        parsed_files = ' '.join(parsed_files.split(' ' + self.START_TOKEN + ' ' + self.END_TOKEN + ' '))

        store_pickle(parsed_files, abspath('out', 'parsed_files_clean_data.pickle'))

    def parse_data(self, parsed_files):
        print('Parsing data...')

        vocab_list = []
        vocab_freq_dict = {}
        index_word_dict = {}
        word_index_dict = {}

        vocab_list = parsed_files.split(' ')
        vocab_freq_dict = dict(Counter(vocab_list))

        for i in range(len(vocab_list)):
            index_word_dict[i] = vocab_list[i]
            word_index_dict[vocab_list[i]] = i

        store_pickle(vocab_list, abspath('out', 'vocab_list.pickle'))
        store_pickle(vocab_freq_dict, abspath('out', 'vocab_freq_dict.pickle'))
        store_pickle(index_word_dict, abspath('out', 'index_word_dict.pickle'))
        store_pickle(word_index_dict, abspath('out', 'word_index_dict.pickle'))

if __name__ == '__main__':
    parser = Parser()