import numpy as np
from src.utils import load_pickle, abspath
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec


class Vectorizer:
    # corpus will be parsed and in the format [[[words] sentences/paragraphs] documents]
    def __init__(self, corpus):
        # word freqs and all could be given from parser
        self.corpus = corpus
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

    def extract_info(self):
        for doc in self.corpus:
            pos_tags = nltk.pos_tag_sents(doc)
            self.corpus_pos_tags.append(pos_tags)
            ner_tags = nltk.ne_chunk_sents(pos_tags)
            self.corpus_ner_tags.append(ner_tags)

            for sent in ner_tags:
                for node in sent:
                    if type(node) == nltk.Tree:
                        # It is a named entity
                        self.all_ner_tags.add(node.label())
                        for word, tag in node:
                            self.all_pos_tags.add(tag)
                            self.word_freq.update([word])
                    else:
                        # extract pos tags
                        self.all_pos_tags.add(node[1])
                        self.word_freq.update([node[0]])

        self.vocabulary = self.word_freq.keys()
        self.index_to_word = dict(enumerate(self.vocabulary))
        self.word_to_index = {v: k for k, v in self.index_to_word.items()}
        self.index_to_pos = dict(enumerate(self.all_pos_tags))
        self.pos_to_index = {v: k for k, v in self.index_to_pos.items()}
        self.index_to_ner = dict(enumerate(self.all_ner_tags))
        self.ner_to_index = {v: k for k, v in self.index_to_ner.items()}

        print("Vocab")
        print(self.vocabulary)
        print("Word freqs")
        print(self.word_freq)
        print("Word to index")
        print(self.word_to_index)
        print("Index to word")
        print(self.index_to_word)
        print("All pos tags")
        print(self.all_pos_tags)
        print("Index to pos")
        print(self.index_to_pos)
        print("Pos to index")
        print(self.pos_to_index)
        print("All ner tags")
        print(self.all_ner_tags)
        print("Index to ner")
        print(self.index_to_ner)
        print("ner to index")
        print(self.ner_to_index)

    def convert_to_vectors(self, min_freq=0):
        word2vec_size = 128
        pos2vec_size = 50
        ner2vec_size = 20

        self.pos_vectors = [np.random.uniform(-1, 1, pos2vec_size) for _ in self.all_pos_tags]
        self.ner_vectors = [np.random.uniform(-1, 1, ner2vec_size) for _ in self.all_ner_tags]

        model = word2vec.Word2Vec([sent for doc in self.corpus for sent in doc], iter=20, min_count=min_freq, size=word2vec_size, workers=4)
        for i in range(len(self.vocabulary)):
            vector = model.wv[self.index_to_word.get(i)]
            if vector is not None:
                self.word_vectors.insert(i, vector)

        print("Pos vectors")
        print(self.pos_vectors)
        print("Ner vectors")
        print(self.ner_vectors)
        print("Word vectors")
        print(self.word_vectors)

    # def generate_corpus_vectors(self):



    def get_tf_idf_vectors(self):
        flat_corpus = [' '.join(sent) for doc in self.corpus for sent in doc]
        freqs = Counter()
        for doc in self.corpus:
            for sent in doc:
                freqs.update(sent)
        # print(flat_corpus)
        vtr = TfidfVectorizer()
        # could also use CountVectors
        vtr.fit(flat_corpus)
        train_vectors = vtr.transform(flat_corpus)
        print(vtr.idf_)
        print(freqs)
        print(vtr.get_feature_names())
        print(vtr.vocabulary_)
        print(train_vectors.shape)
        print(np.array(train_vectors))
        # return train_vectors
        # X_test_vectors = vtr.transform(X_test)
        # print(vtr.get_feature_names())
        # could give own vocabulary

    def get_pos_tags(self):
        pos_tags = nltk.pos_tag_sents([sent for doc in self.corpus for sent in doc])
        print(pos_tags)
        # get all possible pos tags
        # create random vector for all pos tags of size 50
        # for each tag put the corresponding vector
        all_tags = set()
        for sent in pos_tags:
            for word, tag in sent:
                if tag not in all_tags:
                    all_tags.add(tag)
        print(list(enumerate(all_tags)))
        return pos_tags

    def get_ner_tags(self, pos_tags):
        ner_tags = nltk.ne_chunk_sents(pos_tags)
        for line in ner_tags:
            for node in line:
                print(type(node))
                print(node)
                if type(node) == nltk.Tree:
                    print(node.label())
            # print(line.leaves())
            # print(type(line))
            # print(line)
        # if ner tag exists for the word
        return ner_tags

    # def get_word_vectors(self):
    #     model = word2vec.Word2Vec(all_sentences, iter=20, min_count=2, size=128, workers=4)


if __name__ == '__main__':
    cps = [[['This', 'is', 'awesome', 'what', 'is', 'this', 'buddy', '.'], 'Ashton gave Aditya a punch at the Hockey Stadium .'.split(), 'Votercirlce and Micromax merge to create the new Canvas2 .'.split(), ['I', 'am', 'really', 'nervous', 'i', 'dont', 'know', 'what', 'to', 'do', '.']], ['It was raining a lot yesterday and i got completely drenched .'.split(), 'Is there any other way to do this'.split(), 'What is this dude i am so upset please dont do this man'.split()]]
    vectorizer = Vectorizer(cps)
    # vectorizer.get_tf_idf_vectors()
    # pos_tags = vectorizer.get_pos_tags()
    # vectorizer.get_ner_tags(pos_tags)
    # parsed_data = load_pickle(abspath('out', 'parsed_files_clean_data.pickle'))
    # print(parsed_data)
    vectorizer.extract_info()
    vectorizer.convert_to_vectors()
