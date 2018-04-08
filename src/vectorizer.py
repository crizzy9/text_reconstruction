import numpy as np
from src.utils import load_pickle, abspath, store_pickle
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from src.constants import *


class Vectorizer:
    # corpus will be parsed and in the format [[[words] sentences/paragraphs] documents]
    def __init__(self, corpus, min_freq=0):
        # word freqs and all could be given from parser
        self.corpus = corpus
        self.min_freq = min_freq
        self.corpus_pos_tags = []
        self.corpus_ner_tags = []
        self.all_pos_tags = set()
        self.all_ner_tags = set()
        self.word_freq = Counter()
        self.vocabulary = []
        self.word_to_index = {}
        self.index_to_word = {}
        self.pos_to_index = {}
        self.index_to_pos = {}
        self.ner_to_index = {}
        self.index_to_ner = {}
        self.pos_vectors = []
        self.ner_vectors = []
        self.word_vectors = []
        self.features = set()
        self.feature_to_index = {}
        self.index_to_feature = {}
        self.feature_vectors = []
        self.indexed_corpus = []
        self.embedding_matrix = []

    def process_corpus(self):
        for i in range(len(self.corpus)):
            for j in range(len(self.corpus[i])):
                self.corpus[i][j].insert(0, START_TOKEN)
                self.corpus[i][j].append(END_TOKEN)
                self.word_freq.update(self.corpus[i][j])

        # replacing min freq with UNK
        for i in range(len(self.corpus)):
            for j in range(len(self.corpus[i])):
                for k in range(len(self.corpus[i][j])):
                    word = self.corpus[i][j][k]
                    if self.word_freq.get(word) <= self.min_freq:
                        self.corpus[i][j][k] = UNKNOWN_TOKEN

        unknowns = [k for k, v in self.word_freq.items() if v <= self.min_freq]
        self.word_freq.update({UNKNOWN_TOKEN: sum([self.word_freq.get(u) for u in unknowns])})
        for u in unknowns:
            del self.word_freq[u]
        self.vocabulary = list(self.word_freq.keys())
        self.index_to_word = dict(enumerate(self.vocabulary))
        self.word_to_index = {v: k for k, v in self.index_to_word.items()}

        print(self.vocabulary)
        print(self.word_freq)

    def extract_info(self):
        for doc in self.corpus:
            pos_tags = list(nltk.pos_tag_sents(doc))
            self.corpus_pos_tags.append(pos_tags)
            ner_tags = list(nltk.ne_chunk_sents(pos_tags))
            self.corpus_ner_tags.append(ner_tags)
            for sent in ner_tags:
                for node in sent:
                    if type(node) == nltk.Tree:
                        # It is a named entity
                        ner_tag = node.label()
                        self.all_ner_tags.add(ner_tag)
                        for word, tag in node:
                            self.all_pos_tags.add(tag)
                            self.features.add((word, tag, ner_tag))
                    else:
                        # extract pos tags
                        ner_tag = 'None'
                        word = node[0]
                        pos_tag = node[1]
                        self.all_ner_tags.add(ner_tag)
                        self.all_pos_tags.add(pos_tag)
                        self.features.add((word, pos_tag, ner_tag))

        self.index_to_pos = dict(enumerate(self.all_pos_tags))
        self.pos_to_index = {v: k for k, v in self.index_to_pos.items()}
        self.index_to_ner = dict(enumerate(self.all_ner_tags))
        self.ner_to_index = {v: k for k, v in self.index_to_ner.items()}
        self.index_to_feature = dict(enumerate(self.features))
        self.feature_to_index = {v: k for k, v in self.index_to_feature.items()}

        # print("Vocab")
        # print(self.vocabulary)
        # print("Word freqs")
        # print(self.word_freq)
        # print("Word to index")
        # print(self.word_to_index)
        # print("Index to word")
        # print(self.index_to_word)
        # print("All pos tags")
        # print(self.all_pos_tags)
        # print("Index to pos")
        # print(self.index_to_pos)
        # print("Pos to index")
        # print(self.pos_to_index)
        # print("All ner tags")
        # print(self.all_ner_tags)
        # print("Index to ner")
        # print(self.index_to_ner)
        # print("ner to index")
        # print(self.ner_to_index)
        # print("corpus ner tags")
        # for x in self.corpus_ner_tags:
        #     for y in x:
        #         print(y)
        #
        print("Features")
        print(self.features)
        print("features len")
        print(len(self.features))
        # print("Index to feature")
        # print(self.index_to_feature)
        # print("Feature to index")
        # print(self.feature_to_index)

    def convert_to_vectors(self):
        word2vec_size = 128
        pos2vec_size = 50
        ner2vec_size = 20

        # self.pos_vectors = [np.random.uniform(-1, 1, pos2vec_size) for _ in self.all_pos_tags]
        print("len all pos tags")
        n_pos_tags = len(self.all_pos_tags)
        print(n_pos_tags)
        for tag in self.all_pos_tags:
            tag_index = self.pos_to_index.get(tag)
            self.pos_vectors.append([1 if i == tag_index else 0 for i in range(n_pos_tags)])

        # self.ner_vectors = [np.random.uniform(-1, 1, ner2vec_size) for _ in self.all_ner_tags]
        print("len all ner tags")
        n_ner_tags = len(self.all_ner_tags)
        print(n_ner_tags)
        for tag in self.all_ner_tags:
            tag_index = self.ner_to_index.get(tag)
            self.ner_vectors.append([1 if i == tag_index else 0 for i in range(n_ner_tags)])

        model = word2vec.Word2Vec([sent for doc in self.corpus for sent in doc], iter=20, min_count=self.min_freq, size=word2vec_size, workers=4)
        for i in range(len(self.vocabulary)):
            vector = model.wv[self.index_to_word.get(i)]
            if vector is not None:
                self.word_vectors.insert(i, vector)

        # print("Pos vectors")
        # print(self.pos_vectors)
        # print("Ner vectors")
        # print(self.ner_vectors)
        # print("Word vectors")
        # print(self.word_vectors)

    def generate_vectors(self):
        for doc_ner_tags in self.corpus_ner_tags:
            doc_indices = []
            for sent in doc_ner_tags:
                sent_indices = []
                for node in sent:
                    if type(node) == nltk.Tree:
                        ner_tag = node.label()
                        for word, tag in node:
                            # what to do with ngram with ner how to add all pos tags if adding ngram in index?
                            sent_indices.append(self.feature_to_index.get((word, tag, ner_tag)))
                    else:
                        ner_tag = 'None'
                        word = node[0]
                        pos_tag = node[1]
                        sent_indices.append(self.feature_to_index.get((word, pos_tag, ner_tag)))
                doc_indices.append(sent_indices)
            self.indexed_corpus.append(doc_indices)
        print("INDEXED CORPUS")
        print(self.indexed_corpus)

    def create_embedding_matrix(self):
        corpus_size = sum(self.word_freq.values())
        for word, pos_tag, ner_tag in self.features:
            self.embedding_matrix.append(np.concatenate([self.word_vectors[self.word_to_index.get(word)], self.pos_vectors[self.pos_to_index.get(pos_tag)], self.ner_vectors[self.ner_to_index.get(ner_tag)], [self.word_freq.get(word)/corpus_size]]))
        self.embedding_matrix = np.array(self.embedding_matrix)
        print("EMBEDDING MATRIX")
        print(self.embedding_matrix)
        print(self.embedding_matrix.shape)

    def save_data(self):
        store_pickle(self.vocabulary, abspath(OUTPUT_DIR, VOCAB_PICKLE))
        store_pickle(self.feature_to_index, abspath(OUTPUT_DIR, FEATURES_TO_INDEX_PICKLE))
        store_pickle(self.index_to_feature, abspath(OUTPUT_DIR, INDEX_TO_FEATURES_PICKLE))
        store_pickle(self.embedding_matrix, abspath(OUTPUT_DIR, EMBEDDING_MATRIX_PICKLE))
        store_pickle(self.indexed_corpus, abspath(OUTPUT_DIR, INDEX_CORPUS_PICKLE))


if __name__ == '__main__':
    # cps = [[['This', 'is', 'awesome', 'what', 'is', 'this', 'buddy', '.'], 'Ashton gave Aditya a punch at the Hockey Stadium .'.split(), 'Votercirlce and Micromax merge to create the new Canvas2 .'.split(), ['I', 'am', 'really', 'nervous', 'i', 'dont', 'know', 'what', 'to', 'do', '.']], ['It was raining a lot yesterday and i got completely drenched .'.split(), 'Is there any other way to do this'.split(), 'What is this dude i am so upset please dont do this man'.split()]]
    # parsed_data = load_pickle(abspath('out', DATASET_FOLDER + CLEAN_DATA_PICKLE))
    # corpus = parsed_data[:100]
    with open(abspath(DATASET_DIR, 'data', 'train.en'), 'r') as f:
        parsed_data = f.readlines()
    corpus = [sent.split() for sent in parsed_data]
    vectorizer = Vectorizer([corpus], 5)
    vectorizer.process_corpus()
    vectorizer.extract_info()
    vectorizer.convert_to_vectors()
    vectorizer.generate_vectors()
    vectorizer.create_embedding_matrix()
    vectorizer.save_data()