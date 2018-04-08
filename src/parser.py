# Library Imports
from src.utils import *
from collections import Counter
from src.constants import *
from nltk import sent_tokenize


class Parser:

    def __init__(self, train_directory, lowercase=False, min_freq=0):
        self.lowercase = lowercase
        self.min_freq = min_freq
        print('START')

        if not os.path.isfile(abspath(OUTPUT_DIR, FETCH_DATA_PICKLE)):
            self.fetch_data(train_directory)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, FETCH_DATA_PICKLE))

        if not os.path.isfile(abspath(OUTPUT_DIR, CLEAN_DATA_PICKLE)):
            self.clean_data(parsed_files)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, CLEAN_DATA_PICKLE))

        print('END')

    def fetch_data(self, train_directory):
        print('Retrieving data from dataset...')
        train_file_locations_list = [folder[0] for folder in os.walk(train_directory)]  # List of all folders and subfolders from train_directory

        # Get all file locations into a list
        train_file_locations = []
        for train_directory in train_file_locations_list:
            train_file_locations.append([abspath(train_directory, f) for f in os.listdir(train_directory) if os.path.isfile(os.path.join(train_directory, f))])
        train_file_locations = [item for sublist in train_file_locations for item in sublist]  # Flatten out list of lists

        parsed_files = ''
        for file in train_file_locations:
            try:
                with open(file, 'r', errors='ignore') as f:
                    parsed_files += f.read()

            except:
                print(file)
                continue
        store_pickle(parsed_files, abspath(OUTPUT_DIR, FETCH_DATA_PICKLE))

    def clean_data(self, parsed_files):
        print('Cleaning data...')

        if self.lowercase:
            parsed_files = parsed_files.lower()

        # Replace all spaces with a single space
        parsed_files = ' '.join(parsed_files.split())

        # Add START and STOP tags around sentences
        parsed_files = sent_tokenize(parsed_files)
        parsed_files = ' '.join([START_TOKEN + ' ' + parsed_files[i] + ' ' + END_TOKEN for i in range(len(parsed_files))])

        # Replace all tokens with a count of min_freq or less with the out-of-vocabulary symbol UNK.
        parsed_files = ' ' + parsed_files + ' '  # Append spaces for easy replacement
        for key, value in Counter(parsed_files.split(' ')).items():
            if value < self.min_freq:
                parsed_files = parsed_files.replace(' ' + key + ' ', ' ' + UNKNOWN_TOKEN + ' ')
                parsed_files = parsed_files.replace(' ' + key + ' ', ' ' + UNKNOWN_TOKEN + ' ')  # For multiple consecutive occurrences

        # Replace all spaces with a single space
        parsed_files = ' '.join(parsed_files.split())

        # Replace all blank sentences with a single space
        parsed_files = ' '.join(parsed_files.split(' ' + START_TOKEN + ' ' + END_TOKEN + ' '))

        store_pickle(parsed_files, abspath(OUTPUT_DIR, CLEAN_DATA_PICKLE))

if __name__ == '__main__':
    parser = Parser(train_directory=abspath(DATASET_DIR), lowercase=True, min_freq=5)
