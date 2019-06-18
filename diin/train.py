import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from diin.graph import Graph
import tensorflow as tf
from diin import args
from utils.load_data import load_char_word_dynamic_data
import pickle

p_c_index, h_c_index, p_w_index, h_w_index, label = load_char_word_dynamic_data('../input/train.csv', data_size=None)
p_c_index_evl, h_c_index_evl, p_w_index_evl, h_w_index_evl, label_evl = load_char_word_dynamic_data('../input/dev.csv',
                                                                                                    data_size=2000)
p_c_index_holder = tf.placeholder(name='p_index', shape=(None, args.max_char_len), dtype=tf.int32)
h_c_index_holder = tf.placeholder(name='h_index', shape=(None, args.max_char_len), dtype=tf.int32)
p_w_index_holder = tf.placeholder(name='p_vec', shape=(None, args.max_word_len), dtype=tf.int32)
h_w_index_holder = tf.placeholder(name='h_vec', shape=(None, args.max_word_len), dtype=tf.int32)
label_holder = tf.placeholder(name='label', shape=(None,), dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices(
    (p_c_index_holder, h_c_index_holder, p_w_index_holder, h_w_index_holder, label_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with open('../output/word2vec/w2v.vec', 'rb')as file:
    embedding = pickle.load(file)
model = Graph(word_embedding=embedding)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_c_index_holder: p_c_index,
                                              h_c_index_holder: h_c_index,
                                              p_w_index_holder: p_w_index,
                                              h_w_index_holder: h_w_index,
                                              label_holder: label})
    steps = int(len(label) / args.batch_size)
    for epoch in range(args.epochs):
        embedding_train = None
        for step in range(steps):
            try:
                p_index_batch, h_index_batch, p_vec_batch, h_vec_batch, label_batch = sess.run(next_element)
                loss, _, predict, acc, embedding_train = sess.run(
                    [model.loss, model.train_op, model.predict, model.accuracy, model.word_embed],
                    feed_dict={model.p_c: p_index_batch,
                               model.h_c: h_index_batch,
                               model.p_w: p_vec_batch,
                               model.h_w: h_vec_batch,
                               model.y: label_batch,
                               model.keep_prob: args.keep_prob})
                print('epoch:', epoch, ' step:', step, ' loss:', loss, ' acc:', acc)
            except tf.errors.OutOfRangeError:
                print('\n')

        predict, acc, loss = sess.run([model.predict, model.accuracy, model.loss],
                                      feed_dict={model.p_c: p_c_index_evl,
                                                 model.h_c: h_c_index_evl,
                                                 model.p_w: p_w_index_evl,
                                                 model.h_w: h_w_index_evl,
                                                 model.y: label_evl,
                                                 model.keep_prob: 1})
        print('epoch:', epoch, ' evl acc:', acc, ' evl loss:', loss)
        saver.save(sess, f'../output/diin/diin_{epoch}.ckpt')
        with open('../output/diin/w2v.vec', 'wb')as file:
            pickle.dump(embedding_train, file)
            print('save w2v done')
        print('save model done')
        print('\n')
