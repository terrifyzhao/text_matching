import tensorflow as tf
from abcnn import args


class Graph:

    def __init__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.padding = tf.placeholder(dtype=tf.float32, shape=(None, 2, args.char_embedding_size), name='padding')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')

        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
                                         name='embedding')
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        attention_maxtrix = []



        p_embedding = tf.concat((self.padding, p_embedding, self.padding), axis=1)
        h_embedding = tf.concat((self.padding, h_embedding, self.padding), axis=1)

        p_embedding = tf.expand_dims(p_embedding, axis=3)
        h_embedding = tf.expand_dims(h_embedding, axis=3)

        p = tf.layers.conv2d(p_embedding,
                             filters=256,
                             kernel_size=(args.filter_width, args.filter_height),
                             activation='relu')
        h = tf.layers.conv2d(h_embedding,
                             filters=256,
                             kernel_size=(args.filter_width, args.filter_height),
                             activation='relu')

        p = self.dropout(p)
        h = self.dropout(h)

        p_w = tf.layers.average_pooling2d(p, pool_size=(args.filter_width, 1), strides=1)
        h_w = tf.layers.average_pooling2d(h, pool_size=(args.filter_width, 1), strides=1)

        p_w = tf.reshape(p_w, shape=(-1, p_w.shape[1], p_w.shape[2] * p_w.shape[3]))
        h_w = tf.reshape(h_w, shape=(-1, h_w.shape[1], h_w.shape[2] * h_w.shape[3]))

        # p = tf.concat((self.padding, p_w, self.padding), axis=1)
        # h = tf.concat((self.padding, h_w, self.padding), axis=1)

        p = tf.expand_dims(p_w, axis=3)
        h = tf.expand_dims(h_w, axis=3)

        p = tf.layers.conv2d(p,
                             filters=128,
                             kernel_size=(args.filter_width, 256),
                             activation='relu')
        h = tf.layers.conv2d(h,
                             filters=128,
                             kernel_size=(args.filter_width, 256),
                             activation='relu')

        p = self.dropout(p)
        h = self.dropout(h)

        p_all = tf.reduce_mean(p, axis=1)
        h_all = tf.reduce_mean(h, axis=1)

        x = tf.concat((p_all, h_all), axis=2)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))

        # v = tf.layers.dense(x, 256, activation='relu')
        # v = self.dropout(v)
        logits = tf.layers.dense(x, 2, activation='relu')

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
