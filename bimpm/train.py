import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from bimpm.graph import Graph
import tensorflow as tf
from bimpm import args
from utils.load_data_for_bimpm import load_data

p_index, h_index, p_vec, h_vec, label = load_data('../input/train.csv')
p_index_dev, h_index_dev, p_vec_dev, h_vec_dev, label_dev = load_data('../input/dev.csv', data_size=100)
p_index_holder = tf.placeholder(name='p_index', shape=(None, args.max_char_len), dtype=tf.int32)
h_index_holder = tf.placeholder(name='h_index', shape=(None, args.max_char_len), dtype=tf.int32)
p_vec_holder = tf.placeholder(name='p_vec', shape=(None, args.max_word_len, args.word_embedding_len),
                              dtype=tf.float32)
h_vec_holder = tf.placeholder(name='h_vec', shape=(None, args.max_word_len, args.word_embedding_len),
                              dtype=tf.float32)
label_holder = tf.placeholder(name='label', shape=(None,), dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((p_index_holder, h_index_holder, p_vec_holder, h_vec_holder, label_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
model = Graph()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_index_holder: p_index,
                                              h_index_holder: h_index,
                                              p_vec_holder: p_vec,
                                              h_vec_holder: h_vec,
                                              label_holder: label})
    steps = int(len(label) / args.batch_size)
    for epoch in range(args.epochs):
        for step in range(steps):
            try:
                p_index_batch, h_index_batch, p_vec_batch, h_vec_batch, label_batch = sess.run(next_element)
                loss, _, predict, acc = sess.run([model.loss, model.train_op, model.predict, model.accuracy],
                                                 feed_dict={model.p: p_index_batch,
                                                            model.h: h_index_batch,
                                                            model.p_vec: p_vec_batch,
                                                            model.h_vec: h_vec_batch,
                                                            model.y: label_batch,
                                                            model.keep_prob: args.keep_prob})
                print('epoch:', epoch, ' step:', step, ' loss:', loss / args.batch_size, ' acc:', acc)
            except tf.errors.OutOfRangeError:
                print('\n')

        predict, acc = sess.run([model.predict, model.accuracy],
                                feed_dict={model.p: p_index_dev,
                                           model.h: h_index_dev,
                                           model.p_vec: p_vec_dev,
                                           model.h_vec: h_vec_dev,
                                           model.y: label_dev,
                                           model.keep_prob: 1})
        print('epoch:', epoch, ' dev acc:', acc)
        saver.save(sess, f'../output/bimpm/BiMPM_{epoch}.ckpt')
        print('save model done')
        print('\n')
