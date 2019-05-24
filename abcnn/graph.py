import tensorflow as tf
from abcnn import args


class Graph:

    def __init__(self, abcnn1=False, abcnn2=False):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')

        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
                                         name='embedding')

        init_random = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
        self.W0 = tf.get_variable(dtype=tf.float32, shape=(args.seq_length, args.char_embedding_size),
                                  name='W0', initializer=init_random)
        self.W1 = tf.get_variable(dtype=tf.float32, shape=(args.seq_length, args.char_embedding_size),
                                  name='W1', initializer=init_random)

        self.abcnn1 = abcnn1
        self.abcnn2 = abcnn2
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        if self.abcnn1:
            p_embedding = tf.expand_dims(p_embedding, axis=3)
            h_embedding = tf.expand_dims(h_embedding, axis=3)

            attention_matrix = tf.sqrt(tf.reduce_sum(
                tf.square(tf.transpose(p_embedding, perm=[0, 2, 1, 3]) - tf.transpose(h_embedding, perm=[0, 2, 3, 1])),
                axis=1))

            p_embedding = tf.einsum("ijk,kl->ijl", tf.transpose(attention_matrix, perm=[0, 2, 1]), self.W0)
            h_embedding = tf.einsum("ijk,kl->ijl", attention_matrix, self.W1)

        p_embedding = tf.pad(p_embedding, paddings=[[0, 0], [2, 2], [0, 0]])
        h_embedding = tf.pad(h_embedding, paddings=[[0, 0], [2, 2], [0, 0]])

        p_embedding = tf.expand_dims(p_embedding, axis=3)
        h_embedding = tf.expand_dims(h_embedding, axis=3)

        p = tf.layers.conv2d(p_embedding,
                             filters=args.cnn1_filters,
                             kernel_size=(args.filter_width, args.filter_height),
                             activation='relu')
        h = tf.layers.conv2d(h_embedding,
                             filters=args.cnn1_filters,
                             kernel_size=(args.filter_width, args.filter_height),
                             activation='relu')

        p = self.dropout(p)
        h = self.dropout(h)

        if self.abcnn2:
            attention_pool_matrix = tf.sqrt(
                tf.reduce_sum(tf.square(tf.transpose(p, perm=[0, 3, 1, 2]) - tf.transpose(h, perm=[0, 3, 2, 1])),
                              axis=1))
            p_sum = tf.reduce_sum(attention_pool_matrix, axis=2, keep_dims=True)
            h_sum = tf.reduce_sum(attention_pool_matrix, axis=1, keep_dims=True)

            p = tf.reshape(p, shape=(-1, p.shape[1], p.shape[2] * p.shape[3]))
            h = tf.reshape(h, shape=(-1, h.shape[1], h.shape[2] * h.shape[3]))

            p = tf.multiply(p, p_sum)
            h = tf.multiply(h, tf.matrix_transpose(h_sum))
        else:
            p = tf.reshape(p, shape=(-1, p.shape[1], p.shape[2] * p.shape[3]))
            h = tf.reshape(h, shape=(-1, h.shape[1], h.shape[2] * h.shape[3]))

        p = tf.pad(p, paddings=[[0, 0], [2, 2], [0, 0]])
        h = tf.pad(h, paddings=[[0, 0], [2, 2], [0, 0]])

        p = tf.expand_dims(p, axis=3)
        h = tf.expand_dims(h, axis=3)

        p = tf.layers.conv2d(p,
                             filters=args.cnn2_filters,
                             kernel_size=(args.filter_width, args.cnn1_filters),
                             activation='relu')
        h = tf.layers.conv2d(h,
                             filters=args.cnn2_filters,
                             kernel_size=(args.filter_width, args.cnn1_filters),
                             activation='relu')

        p = self.dropout(p)
        h = self.dropout(h)

        p_all = tf.reduce_mean(p, axis=1)
        h_all = tf.reduce_mean(h, axis=1)

        x = tf.concat((p_all, h_all), axis=2)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))

        x = tf.layers.dense(x, 128, activation='relu')
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
