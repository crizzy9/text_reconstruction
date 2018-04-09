import time
import tensorflow as tf
from src.utils import *
from src.constants import *

# import tensorflow.contrib.seq2seq as seq2seq
# from tensorflow.python.ops.rnn_cell import LSTMCell
# from tensorflow.python.ops.rnn_cell import MultiRNNCell
# from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense


class Autoencoder:

    NUM_EPOCHS = 10
    RNN_STATE_DIM = 128
    RNN_NUM_LAYERS = 1
    ENCODER_EMBEDDING_DIM = DECODER_EMBEDDING_DIM = 177

    BATCH_SIZE = 17
    LEARNING_RATE = 0.0003

    def __init__(self):
        self.model_dir = abspath(OUTPUT_DIR, 'model')
        vect_dir = abspath(OUTPUT_DIR, 'vect')
        self.vocabulary = load_pickle(os.path.join(vect_dir, 'vocabulary.pickle'))
        self.src_dictionary = load_pickle(os.path.join(vect_dir, 'features_to_index.pickle'))
        self.src_reverse_dictionary = load_pickle(os.path.join(vect_dir, 'index_to_features.pickle'))
        self.indexed_corpus = load_pickle(os.path.join(vect_dir, 'indexed_corpus.pickle'))
        # only when one document is there
        self.indexed_corpus = self.indexed_corpus[0]
        print("No of sentences", len(self.indexed_corpus))
        self.embedding_matrix = load_pickle(os.path.join(vect_dir, 'embedding_matrix.pickle'))
        print("Embedding matrix shape", self.embedding_matrix.shape)
        self.max_length = max([len(line) for line in self.indexed_corpus])

    def initialize_model(self):
        # encoder_emb_layer = self.embedding_matrix
        # decoder_emb_layer = self.embedding_matrix

        INPUT_NUM_VOCAB = len(self.src_dictionary)
        OUTPUT_NUM_VOCAB = len(self.src_dictionary)

        tf.reset_default_graph()

        self.encoder_input_seq = tf.placeholder(tf.int32, [None, None], name='encoder_input_seq')

        self.encoder_seq_len = tf.placeholder(tf.int32, (None,), name='encoder_seq_len')

        # Decoder placeholders
        self.decoder_output_seq = tf.placeholder(tf.int32, [None, None], name='decoder_output_seq')

        self.decoder_seq_len = tf.placeholder(tf.int32, (None,), name='decoder_seq_len')

        max_decoder_seq_len = tf.reduce_max(self.decoder_seq_len, name='max_decoder_seq_len')

        encoder_input_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.encoder_input_seq)

        encoder_multi_cell = tf.nn.rnn_cell.BasicLSTMCell(self.RNN_STATE_DIM)

        self.encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_multi_cell, encoder_input_embedded, sequence_length=self.encoder_seq_len, dtype=tf.float64)

        decoder_raw_seq = self.decoder_output_seq[:, :-1]

        go_prefixes = tf.fill([self.BATCH_SIZE, 1], self.src_dictionary[('<s>', 'None', 'None')])

        decoder_input_seq = tf.concat([go_prefixes, decoder_raw_seq], 1)

        decoder_input_embedded = tf.nn.embedding_lookup(self.embedding_matrix,
                                                        decoder_input_seq)

        decoder_multi_cell = tf.nn.rnn_cell.BasicLSTMCell(self.RNN_STATE_DIM)

        output_layer_kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
        output_layer = Dense(
            OUTPUT_NUM_VOCAB,
            kernel_initializer=output_layer_kernel_initializer
        )

        with tf.variable_scope("decode"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=decoder_input_embedded,
                sequence_length=[self.max_length for x in range(self.BATCH_SIZE)],
                time_major=False
            )

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_multi_cell,
                training_helper,
                encoder_state,
                output_layer
            )

            training_decoder_output_seq, _, _ = tf.contrib.seq2seq.dynamic_decode(
                training_decoder,
                impute_finished=True,
                maximum_iterations=self.max_length
            )

        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile(
                tf.constant([self.src_dictionary[('<s>', 'None', 'None')]],
                            dtype=tf.int32),
                [self.BATCH_SIZE],
                name='start_tokens')

            # Helper for the inference process.
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding_matrix,
                start_tokens=start_tokens,
                end_token=self.src_dictionary[('</s>', 'None', 'None')]
            )

            # Basic decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_multi_cell,
                inference_helper,
                encoder_state,
                output_layer
            )

            # Perform dynamic decoding using the decoder
            inference_decoder_output_seq, _, _ = tf.contrib.seq2seq.dynamic_decode(
                inference_decoder,
                impute_finished=True,
                maximum_iterations=self.max_length
            )

        training_logits = tf.identity(training_decoder_output_seq.rnn_output, name='logits')
        inference_logits = tf.identity(inference_decoder_output_seq.sample_id, name='predictions')

        # Create the weights for sequence_loss
        masks = tf.sequence_mask(
            self.decoder_seq_len,
            self.max_length,
            dtype=tf.float64,
            name='masks'
        )

        self.cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            self.decoder_output_seq,
            masks
        )

        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train_pred = training_decoder_output_seq.sample_id

        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                            for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

    @staticmethod
    def pad(xs, size, pad):
        return xs + [pad] * (size - len(xs))

    def run(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        cost_data = []
        start = time.time()

        for epoch in range(1, self.NUM_EPOCHS + 1):
            for batch_idx in range(len(self.indexed_corpus) // self.BATCH_SIZE):

                input_batch, input_lengths, output_batch, output_lengths = [], [], [], []
                for sentence in self.indexed_corpus[batch_idx:batch_idx + self.BATCH_SIZE]:
                    symbol_sent = sentence
                    padded_symbol_sent = self.pad(symbol_sent, self.max_length, self.src_dictionary[('<s>', 'None', 'None')])
                    input_batch.append(padded_symbol_sent)
                    input_lengths.append(len(sentence))
                for sentence in self.indexed_corpus[batch_idx:batch_idx + self.BATCH_SIZE]:
                    symbol_sent = sentence
                    padded_symbol_sent = self.pad(symbol_sent, self.max_length, self.src_dictionary[('</s>', 'None', 'None')])
                    output_batch.append(padded_symbol_sent)
                    output_lengths.append(len(sentence))

                _, cost_val, pred, encoder_out = sess.run(
                    [self.train_op, self.cost, self.train_pred, self.encoder_output],
                    feed_dict={
                        self.encoder_input_seq: input_batch,
                        self.encoder_seq_len: input_lengths,
                        self.decoder_output_seq: output_batch,
                        self.decoder_seq_len: output_lengths
                    }
                )

                if batch_idx % 250 == 0 or batch_idx == len(self.indexed_corpus) // self.BATCH_SIZE - 1:
                    print('Epcoh {}. Batch {}/{}. Cost {}'.format(epoch, batch_idx, len(self.indexed_corpus) // self.BATCH_SIZE,
                                                                  cost_val))
                    print("original", [self.src_reverse_dictionary[i][0] for i in output_batch[0]])
                    print("pred", [self.src_reverse_dictionary[i][0] for i in pred[0]])
                    # print("encoder embeddings ", encoder_out)
                    cost_data.append(self.cost)

            if epoch % 5 == 0:
                self.save_model(sess, epoch, cost_data)

        end = time.time()
        print("time:", end - start)
        sess.close()

    def save_model(self, sess, epoch, cost_data):
        print("saving model")
        self.saver.save(sess, os.path.join(self.model_dir, "model_{}.ckpt".format(epoch)))
        with open(os.path.join(self.model_dir, "loss_{}.txt".format(epoch), "w")) as fp:
            fp.write(str(cost_data))


if __name__ == '__main__':
    ae = Autoencoder()
    ae.initialize_model()
    ae.run()
