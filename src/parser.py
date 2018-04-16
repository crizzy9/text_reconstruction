# Library Imports
import re
from src.utils import *
from collections import Counter
from random import shuffle
from src.constants import *
from nltk import sent_tokenize, word_tokenize


class Parser:

    def __init__(self, train_directory, lowercase=False, paragraphs=False, no_of_files=1000, max_sents=5, min_sents=1, max_words=100, min_words=15, limit=1000000):
        self.lowercase = lowercase
        print('START')

        if not os.path.isfile(abspath(OUTPUT_DIR, FETCH_DATA_PICKLE)):
            self.fetch_data(train_directory, no_of_files)
        parsed_files = load_pickle(abspath(OUTPUT_DIR, FETCH_DATA_PICKLE))

        if not os.path.isfile(abspath(OUTPUT_DIR, CLEAN_DATA_PICKLE)):
            self.clean_data(parsed_files, paragraphs, max_sents, min_sents, max_words, min_words, limit)
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
        shuffle(train_file_locations)
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

    def clean_data(self, parsed_files, paragraphs, max_sents, min_sents, max_words, min_words, limit):
        print('Cleaning data...', len(parsed_files))
        if self.lowercase:
            parsed_files = [file.lower() for file in parsed_files]

        # Replace all spaces with a single space
        if not paragraphs:
            parsed_files = [' '.join(file.split()) for file in parsed_files]
        else:
            parsed_files = [[' '.join(para.split()) for para in re.sub(r'([\n]{2,})', '\n\n', file.strip()).split('\n\n')] for file in parsed_files]
            total_no_paras = sum([len(file) for file in parsed_files])
            avg_paras_per_file = total_no_paras/len(parsed_files)
            print("Para example\n", parsed_files[0])
            print("Total number of paras: {}, Avg paras per file: {}".format(total_no_paras, avg_paras_per_file))

        # Add START and STOP tags around sentences
        data = []
        file_count = 0
        for file in parsed_files:
            file_count += 1
            print("Tokenizing file: {}/{}\t".format(file_count, len(parsed_files)))
            if paragraphs:
                count = 0
                for para in file:
                    sents = sent_tokenize(para)

                    if len(sents) <= max_sents:
                        words = word_tokenize(para)
                        if len(sents) <= min_sents and len(words) < min_words:
                            continue
                        count += 1
                        data.append(words)
                print("Para count for file {} < {}".format(count, max_sents))
                if file_count == 1:
                    print("CONTENTS")
                    print(data)
            else:
                for sent in sent_tokenize(file):
                    words = word_tokenize(sent)
                    if len(words) <= max_words:
                        data.append(words)

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

        print("Total no of {} < {} = {}".format('paragraphs' if paragraphs else 'sentences', max_sents if paragraphs else max_words, len(data)))
        data = data[:limit]
        print("Sample data\n")
        print(data[:3])
        store_pickle(data, abspath(OUTPUT_DIR, CLEAN_DATA_PICKLE))

if __name__ == '__main__':
    parser = Parser(train_directory=abspath(DATASET_DIR), lowercase=False, paragraphs=True, no_of_files=4000)
