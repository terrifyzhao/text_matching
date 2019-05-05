import tensorflow as tf
from convnet import args


class Graph:

    def __init__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')

        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
                                         name='embedding')
        self.M = tf.Variable(tf.random_normal(shape=(args.n_filter, args.n_filter), mean=0, stddev=1),
                             name='M',
                             dtype=tf.float32)
        self.x_feat = tf.Variable(tf.random_normal(shape=(args.batch_size, args.n_filter), mean=0, stddev=1),
                                  name='x_feat',
                                  dtype=tf.float32)

        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        p_embedding = tf.expand_dims(p_embedding, axis=3)
        h_embedding = tf.expand_dims(h_embedding, axis=3)
        p = tf.layers.conv2d(p_embedding,
                             filters=args.n_filter,
                             kernel_size=(args.filter_width, args.filter_height),
                             activation='relu')
        h = tf.layers.conv2d(h_embedding,
                             filters=args.n_filter,
                             kernel_size=(args.filter_width, args.filter_height),
                             activation='relu')

        pool_width = args.seq_length + 1 - args.filter_width
        p_max = tf.layers.max_pooling2d(p, pool_size=(pool_width, 1), strides=1)
        h_max = tf.layers.max_pooling2d(h, pool_size=(pool_width, 1), strides=1)

        p_max = tf.squeeze(tf.squeeze(p_max, axis=1), axis=1)
        h_max = tf.squeeze(tf.squeeze(h_max, axis=1), axis=1)

        # sim = tf.matmul(tf.matmul(p_max, self.M), tf.transpose(h_max))
        sim = tf.matmul(p_max, self.M)
        sim = tf.reduce_sum(tf.multiply(sim, h_max), axis=1, keep_dims=True)

        x = tf.concat((p_max, h_max, sim), axis=1)

        v = tf.layers.dense(x, 256, activation='relu')
        v = self.dropout(v)
        logits = tf.layers.dense(v, 2, activation='relu')

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
