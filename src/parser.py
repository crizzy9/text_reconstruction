# Library Imports
import os
from src.utils import store_pickle, load_pickle, abspath
from collections import Counter
from src.constants import *


class Parser:

    def __init__(self, train_directory, lowercase=False, min_freq=0):
        self.lowercase = lowercase
        self.min_freq = min_freq
        print('Enter init...')

        if not os.path.isfile(abspath(OUTPUT_DIR, 'parsed_files_fetch_data.pickle')):
            self.fetch_data(train_directory)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, 'parsed_files_fetch_data.pickle'))

        if not os.path.isfile(abspath(OUTPUT_DIR, 'parsed_files_clean_data.pickle')):
            self.clean_data(parsed_files)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, 'parsed_files_clean_data.pickle'))

    def fetch_data(self, train_directory):
        print('Retrieving data from dataset...')
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

if __name__ == '__main__':
    train_directory = abspath(DATASET_DIR, 'Gutenberg', 'txt')
    parser = Parser(train_directory, lowercase=True, min_freq=5)
