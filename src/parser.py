# Library Imports
from src.utils import *
from collections import Counter
from src.constants import *
from nltk import sent_tokenize, word_tokenize


class Parser:

    def __init__(self, train_directory, lowercase=False, min_freq=0, no_of_files=1000, max_sen_len=100, no_sent_limit=500000):
        self.lowercase = lowercase
        self.min_freq = min_freq
        print('START')

        if not os.path.isfile(abspath(OUTPUT_DIR, FETCH_DATA_PICKLE)):
            self.fetch_data(train_directory, no_of_files)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, FETCH_DATA_PICKLE))

        if not os.path.isfile(abspath(OUTPUT_DIR, CLEAN_DATA_PICKLE)):
            self.clean_data(parsed_files, max_sen_len, no_sent_limit)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, CLEAN_DATA_PICKLE))

        print('END')

    def fetch_data(self, train_directory, no_of_files):
        print('Retrieving data from dataset...')
        train_file_locations_list = [folder[0] for folder in os.walk(train_directory)]  # List of all folders and subfolders from train_directory

        # Get all file locations into a list
        train_file_locations = []
        for train_directory in train_file_locations_list:
            train_file_locations.append([abspath(train_directory, f) for f in os.listdir(train_directory) if os.path.isfile(os.path.join(train_directory, f))])
        train_file_locations = [item for sublist in train_file_locations for item in sublist]  # Flatten out list of lists
        train_file_locations = train_file_locations[:no_of_files]
        parsed_files = []
        for file in train_file_locations:
            try:
                with open(file, 'r', errors='ignore') as f:
                    parsed_files.append(f.read())

            except:
                print(file)
                continue
        store_pickle(parsed_files, abspath(OUTPUT_DIR, FETCH_DATA_PICKLE))

    def clean_data(self, parsed_files, max_sen_len, no_sent_limit):
        print('Cleaning data...', len(parsed_files))
        if self.lowercase:
            parsed_files = [file.lower() for file in parsed_files]

        # Replace all spaces with a single space
        parsed_files = [' '.join(file.split()) for file in parsed_files]

        # Add START and STOP tags around sentences
        all_sentences = []
        file_count = 0
        for file in parsed_files:
            file_count += 1
            print("Tokenizing file:", file_count)
            for sent in sent_tokenize(file):
                words = word_tokenize(sent)
                if len(words) < max_sen_len:
                    all_sentences.append(words)

        # parsed_files = ' '.join([START_TOKEN + ' ' + parsed_files[i] + ' ' + END_TOKEN for i in range(len(parsed_files))])

        # Replace all tokens with a count of min_freq or less with the out-of-vocabulary symbol UNK.
        # parsed_files = ' ' + parsed_files + ' '  # Append spaces for easy replacement
        # for key, value in Counter(parsed_files.split(' ')).items():
        #     if value < self.min_freq:
        #         parsed_files = parsed_files.replace(' ' + key + ' ', ' ' + UNKNOWN_TOKEN + ' ')
        #         parsed_files = parsed_files.replace(' ' + key + ' ', ' ' + UNKNOWN_TOKEN + ' ')  # For multiple consecutive occurrences

        # Replace all spaces with a single space
        # parsed_files = ' '.join(parsed_files.split())

        # Replace all blank sentences with a single space
        # parsed_files = ' '.join(parsed_files.split(' ' + START_TOKEN + ' ' + END_TOKEN + ' '))

        print("Total no of sentences < {} = {}".format(max_sen_len, len(all_sentences)))
        all_sentences = all_sentences[:no_sent_limit]
        store_pickle(all_sentences, abspath(OUTPUT_DIR, CLEAN_DATA_PICKLE))

if __name__ == '__main__':
    parser = Parser(train_directory=abspath(DATASET_DIR), lowercase=True, min_freq=5)
