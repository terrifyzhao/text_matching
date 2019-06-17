import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from drcn.graph import Graph
import tensorflow as tf
from drcn import args
from utils.load_data import load_all_data
import pickle

p_c_index, h_c_index, p_w_index, h_w_index, p_w_vec, h_w_vec, same_word, label = load_all_data('../input/train.csv',
                                                                                               data_size=100)
p_c_index_evl, h_c_index_evl, p_w_index_evl, h_w_index_evl, p_w_vec_evl, h_w_vec_evl, same_word_evl, label_evl = load_all_data(
    '../input/dev.csv', data_size=200)

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

with open('../output/word2vec/w2v.vec', 'rb')as file:
    embedding = pickle.load(file)
model = Graph(word_embedding=embedding)
saver = tf.train.Saver(max_to_keep=10)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0

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
    steps = int(len(label) / args.batch_size)
    for epoch in range(args.epochs):
        for step in range(steps):
            try:
                p_c_index_batch, h_c_index_batch, p_w_index_batch, h_w_index_batch, p_w_vec_batch, h_w_vec_batch, same_word_batch, label_batch = sess.run(
                    next_element)
                loss, _, predict, acc = sess.run(
                    [model.loss, model.train_op, model.predict, model.accuracy],
                    feed_dict={model.p_c_index: p_c_index_batch,
                               model.h_c_index: h_c_index_batch,
                               model.p_w_index: p_w_index_batch,
                               model.h_w_index: h_w_index_batch,
                               model.p_w_vec: p_w_vec_batch,
                               model.h_w_vec: h_w_vec_batch,
                               model.same_word: same_word_batch,
                               model.y: label_batch,
                               model.keep_prob_embed: args.keep_prob_embed,
                               model.keep_prob_fully: args.keep_prob_fully,
                               model.keep_prob_ae: args.keep_prob_ae,
                               model.bn_training: True})
                print('epoch:', epoch, ' step:', step, ' loss:', loss, ' acc:', acc)
            except tf.errors.OutOfRangeError:
                print('\n')

        predict, acc = sess.run([model.predict, model.accuracy],
                                feed_dict={model.p_c_index: p_c_index_evl,
                                           model.h_c_index: h_c_index_evl,
                                           model.p_w_index: p_w_index_evl,
                                           model.h_w_index: h_w_index_evl,
                                           model.p_w_vec: p_w_vec_evl,
                                           model.h_w_vec: h_w_vec_evl,
                                           model.same_word: same_word_evl,
                                           model.y: label_evl,
                                           model.keep_prob_embed: 1,
                                           model.keep_prob_fully: 1,
                                           model.keep_prob_ae: 1,
                                           model.bn_training: False})
        print('epoch:', epoch, ' dev acc:', acc)
        saver.save(sess, f'../output/diin/diin_{epoch}.ckpt')
        print('save model done')
        print('\n')
