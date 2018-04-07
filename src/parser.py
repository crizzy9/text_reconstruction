# Library Imports
import os
# from os import listdir
# from os.path import isfile, join
from src.utils import store_pickle, load_pickle, abspath
from nltk import sent_tokenize, word_tokenize
from collections import Counter
from src.constants import *


class Parser:

    # take filepaths as variable instead of calculating them in fetch data
    # put constants in constants.py and use them
    # store file locations in variable
    # add start and end tokens after word tokenize
    def __init__(self, filepaths, lowercase=False, min_freq=0):
        self.lowercase = lowercase
        self.min_freq = min_freq
        print('Enter init...')

        if not os.path.isfile(abspath(OUTPUT_DIR, 'parsed_files_fetch_data.pickle')):
            self.fetch_data()
        parsed_files = load_pickle(abspath(OUTPUT_DIR, 'parsed_files_fetch_data.pickle'))

        if not os.path.isfile(abspath(OUTPUT_DIR, 'parsed_files_clean_data.pickle')):
            self.clean_data(parsed_files)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, 'parsed_files_clean_data.pickle'))

        if not os.path.isfile(abspath(OUTPUT_DIR, 'vocab_list.pickle')):
            self.parse_data(parsed_files)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, 'parsed_files.pickle'))
        vocab_list = load_pickle(abspath(OUTPUT_DIR, 'vocab_list.pickle'))
        vocab_freq_dict = load_pickle(abspath(OUTPUT_DIR, 'vocab_freq_dict.pickle'))
        index_word_dict = load_pickle(abspath(OUTPUT_DIR, 'index_word_dict.pickle'))
        word_index_dict = load_pickle(abspath(OUTPUT_DIR, 'word_index_dict.pickle'))

    def fetch_data(self):
        print('Retrieving data from Gutenberg corpus...')
        train_directory = abspath(DATASET_DIR, 'Gutenberg', 'txt')
        # train_file_locations = [os.path.join(train_directory, f) for f in listdir(train_directory) if
        #                        isfile(join(train_directory, f)) and f != '.fuse_hidden0000465600000001']
        # no need to check if file is there ??? doing it so that we don't grab contents of that file
        train_file_locations = [os.path.join(train_directory, f) for f in os.listdir(train_directory)]
        parsed_files = ''
        i = 0
        for file in train_file_locations:
            if i == 10:
                break
            i += 1
            with open(file, 'r', errors='ignore') as f:
                parsed_files += f.read()

        store_pickle(parsed_files, abspath(OUTPUT_DIR, 'parsed_files_fetch_data.pickle'))

    def clean_data(self, parsed_files):
        print('Cleaning data...')

        # Convert to lower case if Modes.lowercase is 1
        if self.lowercase:
            parsed_files = parsed_files.lower()

        # Replace all spaces with a single space
        parsed_files = ' '.join(parsed_files.split())

        # Replace all tokens with a count of 10 or less with the out-of-vocabulary symbol UNK.
        parsed_files = ' ' + parsed_files + ' '  # Append spaces for easy replacement
        for key, value in Counter(parsed_files.split(' ')).items():
            if value < self.min_freq:
                parsed_files = parsed_files.replace(' ' + key + ' ', ' ' + UNKNOWN_TOKEN + ' ')
                parsed_files = parsed_files.replace(' ' + key + ' ', ' ' + UNKNOWN_TOKEN + ' ')  # For multiple consecutive occurrences

        # Replace all spaces with a single space
        parsed_files = ' '.join(parsed_files.split())

        # Replace all blank sentences with a single space
        parsed_files = ' '.join(parsed_files.split(' ' + START_TOKEN + ' ' + END_TOKEN + ' '))

        store_pickle(parsed_files, abspath(OUTPUT_DIR, 'parsed_files_clean_data.pickle'))

    def parse_data(self, parsed_files):
        print('Parsing data...')

        vocab_list = []
        vocab_freq_dict = {}
        index_word_dict = {}
        word_index_dict = {}

        # Append start and end tags and obtain entire vocabulary as a list
        vocab_list.append(START_TOKEN)
        for word in word_tokenize(parsed_files):
            vocab_list.append(word)
            if word in FULL_STOPS:
                vocab_list.append(END_TOKEN)
                vocab_list.append(START_TOKEN)
        vocab_list.append(END_TOKEN)

        # Get list of list of list of words
        parsed_files = sent_tokenize(parsed_files)
        for i in range(len(parsed_files)):
            parsed_files[i] = word_tokenize(parsed_files[i])
        print(parsed_files)

        vocab_freq_dict = dict(Counter(vocab_list))

        for i in range(len(vocab_list)):
            index_word_dict[i] = vocab_list[i]
            word_index_dict[vocab_list[i]] = i

        store_pickle(parsed_files, abspath(OUTPUT_DIR, 'parsed_files.pickle'))
        store_pickle(vocab_list, abspath(OUTPUT_DIR, 'vocab_list.pickle'))
        store_pickle(vocab_freq_dict, abspath(OUTPUT_DIR, 'vocab_freq_dict.pickle'))
        store_pickle(index_word_dict, abspath(OUTPUT_DIR, 'index_word_dict.pickle'))
        store_pickle(word_index_dict, abspath(OUTPUT_DIR, 'word_index_dict.pickle'))




if __name__ == '__main__':
    filepaths = []
    parser = Parser(filepaths, lowercase=True, min_freq=5)