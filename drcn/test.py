import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from drcn.graph import Graph
import tensorflow as tf
from drcn import args
from utils.load_data import load_all_data
import numpy as np
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

p_c_index, h_c_index, p_w_index, h_w_index, p_w_vec, h_w_vec, same_word, label = load_all_data('../input/test.csv',
                                                                                               data_size=None)
p_c_index_holder = tf.placeholder(name='p_c_index', shape=(None, args.max_char_len), dtype=tf.int32)
h_c_index_holder = tf.placeholder(name='h_c_index', shape=(None, args.max_char_len), dtype=tf.int32)
p_w_index_holder = tf.placeholder(name='p_w_index', shape=(None, args.max_word_len), dtype=tf.int32)
h_w_index_holder = tf.placeholder(name='h_w_index', shape=(None, args.max_word_len), dtype=tf.int32)
p_w_vec_holder = tf.placeholder(name='p_w_vec', shape=(None, args.max_word_len, args.word_embedding_len),
                                dtype=tf.float32)
h_w_vec_holder = tf.placeholder(name='h_w_vec', shape=(None, args.max_word_len, args.word_embedding_len),
                                dtype=tf.float32)
same_word_holder = tf.placeholder(name='same_word', shape=(None,), dtype=tf.int32)
label_holder = tf.placeholder(name='y', shape=(None,), dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices(
    (p_c_index_holder, h_c_index_holder, p_w_index_holder, h_w_index_holder, p_w_vec_holder, h_w_vec_holder,
     same_word_holder, label_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with open('../output/drcn/w2v.vec', 'rb')as file:
    embedding = pickle.load(file)
model = Graph(word_embedding=embedding)
saver = tf.train.Saver(max_to_keep=10)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_c_index_holder: p_c_index,
                                              h_c_index_holder: h_c_index,
                                              p_w_index_holder: p_w_index,
                                              h_w_index_holder: h_w_index,
                                              p_w_vec_holder: p_w_vec,
                                              h_w_vec_holder: h_w_vec,
                                              same_word_holder: same_word,
                                              label_holder: label})
    saver.restore(sess, "../output/drcn/drcn37.ckpt")
    steps = int(len(label) / args.batch_size)
    loss_all = []
    acc_all = []
    for step in range(steps):
        try:
            p_c_index_batch, h_c_index_batch, p_w_index_batch, h_w_index_batch, p_w_vec_batch, h_w_vec_batch, same_word_batch, label_batch = sess.run(
                next_element)
            loss, predict, acc = sess.run([model.loss, model.predict, model.accuracy],
                                          feed_dict={model.p_c_index: p_c_index_batch,
                                                     model.h_c_index: h_c_index_batch,
                                                     model.p_w_index: p_w_index_batch,
                                                     model.h_w_index: h_w_index_batch,
                                                     model.p_w_vec: p_w_vec_batch,
                                                     model.h_w_vec: h_w_vec_batch,
                                                     model.same_word: same_word_batch,
                                                     model.y: label_batch,
                                                     model.keep_prob_embed: 1,
                                                     model.keep_prob_fully: 1,
                                                     model.keep_prob_ae: 1,
                                                     model.bn_training: False})
            loss_all.append(loss)
            acc_all.append(acc)
        except tf.errors.OutOfRangeError:
            print('\n')

    loss = np.mean(loss_all)
    acc = np.mean(acc_all)
    print('test loss:', loss, ' test acc:', acc)
