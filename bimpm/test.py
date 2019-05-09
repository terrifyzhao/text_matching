from BiMPM import Graph
import tensorflow as tf
import args
from data_process import load_data
import numpy as np

p_index, h_index, p_vec, h_vec, label = load_data('input/test.csv')
p_index_dev, h_index_dev, p_vec_dev, h_vec_dev, label_dev = load_data('input/dev.csv')
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
config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_index_holder: p_index,
                                              h_index_holder: h_index,
                                              p_vec_holder: p_vec,
                                              h_vec_holder: h_vec,
                                              label_holder: label})
    saver.restore(sess, "/output/BiMPM_0.ckpt")
    steps = int(len(label) / args.batch_size)
    loss_all = []
    acc_all = []
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
            loss_all.append(loss)
            acc_all.append(acc)
        except tf.errors.OutOfRangeError:
            print('\n')

    loss = np.mean(loss_all)
    acc = np.mean(acc_all)
    print('test loss:', loss, ' test acc:', acc)
