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

        if not os.path.isfile(abspath(OUTPUT_DIR, DATASET_FOLDER + FETCH_DATA_PICKLE)):
            self.fetch_data(train_directory)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, DATASET_FOLDER + FETCH_DATA_PICKLE))

        if not os.path.isfile(abspath(OUTPUT_DIR, DATASET_FOLDER + CLEAN_DATA_PICKLE)):
            self.clean_data(parsed_files)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, DATASET_FOLDER + CLEAN_DATA_PICKLE))

    def fetch_data(self, train_directory):
        print('Retrieving data from dataset...')
        train_file_locations = [os.path.join(train_directory, f) for f in os.listdir(train_directory)]
        parsed_files = ''
        for file in train_file_locations:
            # file = file.replace('\'', '\\\'') if '\'' in file else file   # Replace ' with \' in file for open(file)

            try:
                with open(file, 'r', errors='ignore') as f:
                    parsed_files += f.read()

            except:
                print(file)
                continue

        store_pickle(parsed_files, abspath(OUTPUT_DIR, DATASET_FOLDER + FETCH_DATA_PICKLE))

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

        store_pickle(parsed_files, abspath(OUTPUT_DIR, DATASET_FOLDER + CLEAN_DATA_PICKLE))

if __name__ == '__main__':
    train_directory = abspath(DATASET_DIR, DATASET_FOLDER, 'txt')   # Specify the training directory here
    parser = Parser(train_directory, lowercase=True, min_freq=5)
