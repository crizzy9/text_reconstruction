import pickle
import time
import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab
from collections import Counter
import csv
import nltk

# Seq2Seq Items
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import lookup_ops

data_dir = './data3/'

with open(data_dir + "embeddings.pickle", "rb") as fp:
  embeddings = pickle.load(fp)

print("Got the embeddings!")

vocab_size = embeddings.shape[0]
sos_id = embeddings.shape[0]-2
eos_id = embeddings.shape[0]-1
batch_size = 10
tf.reset_default_graph()

reverse_vocab = tf.contrib.lookup.index_to_string_table_from_file(data_dir+"new_vocab.txt")
vocab = lookup_ops.index_table_from_file(data_dir+"new_vocab.txt", default_value=0)
src_dataset = tf.data.TextLineDataset(data_dir+"text.txt")
dataset = tf.data.Dataset.zip((src_dataset, src_dataset))

# string to token
dataset = dataset.map(
    lambda src, tgt: (
        tf.string_split([src]).values, tf.string_split([tgt]).values),
    num_parallel_calls=2)

# word to index
dataset = dataset.map(
    lambda src, tgt: (tf.cast(vocab.lookup(src), tf.int32),
                      tf.cast(vocab.lookup(tgt), tf.int32)),
    num_parallel_calls=2)

dataset = dataset.map(
    lambda src, tgt: (src,
                      tf.concat(([sos_id], tgt), 0),
                      tf.concat((tgt, [eos_id]), 0)),
    num_parallel_calls=2)


# add length
dataset = dataset.map(
    lambda src, target_input, summary: (
        src, target_input, summary, tf.size(src), tf.size(summary)),
    num_parallel_calls=2)


def batching_func(x):
    return x.padded_batch(
        batch_size,  # batch size
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([])),  # src_len
        padding_values=(
            eos_id,  # src
            eos_id,  # src
            eos_id,  # src
            0,
            0))  # len

dataset = batching_func(dataset)


iterator = dataset.make_initializable_iterator()
(inputs_index, target_input, label_index, input_sequence_length, labels_sequence_length) = iterator.get_next()

emb_mat = tf.constant(embeddings)


out_dir = './output/'

with open(out_dir+"{}_labels.txt".format(5), "w") as fp:
    for line in ["Comfortably numb", "Welcome to the machine", "Coming back to life"]:
        fp.write("%s\n" % line)
    print("File saved")

print("Starting session!")
with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer, feed_dict=None)

    ########################## Encoder #######################

    embedded_inputs = tf.nn.embedding_lookup(
        emb_mat, inputs_index)

    embedded_labels = tf.nn.embedding_lookup(
        emb_mat, target_input)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=input_sequence_length,
        inputs=embedded_inputs)


    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        64, encoder_output, memory_sequence_length=input_sequence_length, dtype=tf.float64)

    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)

    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell,
        attention_mechanism,
        attention_layer_size=64,
        alignment_history=False,
        output_attention=True,
        name="attention")

    decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float64).clone(
        cell_state=encoder_state)

    projection_layer = Dense(units=vocab_size, use_bias=False)

    helper = tf.contrib.seq2seq.TrainingHelper(
        embedded_labels, [tf.reduce_max(labels_sequence_length) for _ in range(batch_size)]
        , time_major=False)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        decoder_cell, helper, decoder_initial_state,
        output_layer=projection_layer)


    outputs, output_states, output_seq_length = tf.contrib.seq2seq.dynamic_decode(
        decoder, output_time_major=False,
        swap_memory=False
    )


    logits = outputs.rnn_output

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_index, logits=logits)
    train_loss = (tf.reduce_sum(crossent
                                * tf.sequence_mask(labels_sequence_length, dtype=logits.dtype)) /
                  batch_size)

    global_step = tf.Variable(0, trainable=False)
    inc_gstep = tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(
        0.03, global_step, decay_steps=10, decay_rate=0.9, staircase=True)

    adam_optimizer = tf.train.AdamOptimizer(learning_rate)

    adam_gradients, v = zip(*adam_optimizer.compute_gradients(train_loss))
    adam_gradients, _ = tf.clip_by_global_norm(adam_gradients, 25.0)
    adam_optimize = adam_optimizer.apply_gradients(zip(adam_gradients, v))
    train_prediction = outputs.sample_id


    sess.run(tf.global_variables_initializer())

    average_loss = 0
    for epoch in range(100):
        epoch_start = time.time()
        sess.run(iterator.initializer, feed_dict=None)
        average_loss = 0
        encoder_out_list = []
        labels = []
        predictions = []
        losses = []
        for step in range(5000):
            _, l, pred, t_i, o_i, enc_out = sess.run([adam_optimize, train_loss, train_prediction, target_input, label_index, encoder_output],
                                            feed_dict=None)
            if epoch%5 == 0 or epoch == 0:
                x = reverse_vocab.lookup(tf.constant(pred, tf.int64))
                y = reverse_vocab.lookup(tf.constant(o_i, tf.int64))

        
                for i in enc_out:
                   encoder_out_list.append(i)

                for i in sess.run(y):                   
                    labels.append(i)
                    
                for i in sess.run(x):
                    predictions.append(i)

                for i in sess.run(x):
                    losses.append(l)
                
               
            average_loss += l;
            
            if step == 0 or step %1000 == 0:
                print("step {} loss:: {}".format(step, l))
            if step % 100 == 0:
                print(".", step)

      
        if len(labels) != 0:
            with open(out_dir+"{}_labels.txt".format(epoch), "w") as fp:
                for line in labels:
                    fp.write("%s\n" % line)
            labels = []
            with open(out_dir+"{}_predictions.txt".format(epoch), "w") as fp:
                for line in predictions:
                    fp.write("%s\n" % line)
            predictions = []
            with open(out_dir+"{}_encoder_output.txt".format(epoch), "w") as fp:
                for line in encoder_out_list:
                    fp.write("%s\n" % line)
            encoder_out_list = []
            with open(out_dir+"{}_losses.txt".format(epoch), "w") as fp:
                for line in losses:
                    fp.write("%s\n" % line)
            
        saver = tf.train.Saver()
        save_path = saver.save(sess, out_dir+"model.ckpt")
        print("Current Epoch::", epoch, "average loss::", average_loss / 1000, "time_taken::", time.time()-epoch_start)
