from dssm.graph import Graph
import tensorflow as tf
from utils.load_data import load_data
import args

p, h, y = load_data('input/train.csv', data_size=1000)

model = Graph()
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(args.epochs):
        _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                                feed_dict={model.p: p,
                                           model.h: h,
                                           model.y: y})
        print('loss: ', loss, ' acc:', acc)
