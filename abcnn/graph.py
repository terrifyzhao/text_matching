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

        self.W0 = tf.get_variable(name="aW",
                                  shape=(args.seq_length + 4, args.char_embedding_size),
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004))

        self.abcnn1 = abcnn1
        self.abcnn2 = abcnn2
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")
        return dot_products / (norm1 * norm2)

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        p_embedding = tf.expand_dims(p_embedding, axis=-1)
        h_embedding = tf.expand_dims(h_embedding, axis=-1)

        p_embedding = tf.pad(p_embedding, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])
        h_embedding = tf.pad(h_embedding, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])

        if self.abcnn1:
            euclidean = tf.sqrt(tf.reduce_sum(
                tf.square(tf.transpose(p_embedding, perm=[0, 2, 1, 3]) - tf.transpose(h_embedding, perm=[0, 2, 3, 1])),
                axis=1) + 1e-6)

            attention_matrix = 1 / (euclidean + 1)

            p_attention = tf.expand_dims(tf.einsum("ijk,kl->ijl", attention_matrix, self.W0), -1)
            h_attention = tf.expand_dims(
                tf.einsum("ijk,kl->ijl", tf.transpose(attention_matrix, perm=[0, 2, 1]), self.W0), -1)

            p_embedding = tf.concat([p_embedding, p_attention], axis=-1)
            h_embedding = tf.concat([h_embedding, h_attention], axis=-1)

        p = tf.layers.conv2d(p_embedding,
                             filters=args.cnn1_filters,
                             kernel_size=(args.filter_width, args.filter_height))
        h = tf.layers.conv2d(h_embedding,
                             filters=args.cnn1_filters,
                             kernel_size=(args.filter_width, args.filter_height))

        p = self.dropout(p)
        h = self.dropout(h)

        if self.abcnn2:
            attention_pool_euclidean = tf.sqrt(
                tf.reduce_sum(tf.square(tf.transpose(p, perm=[0, 3, 1, 2]) - tf.transpose(h, perm=[0, 3, 2, 1])),
                              axis=1))
            attention_pool_matrix = 1 / (attention_pool_euclidean + 1)
            p_sum = tf.reduce_sum(attention_pool_matrix, axis=2, keep_dims=True)
            h_sum = tf.reduce_sum(attention_pool_matrix, axis=1, keep_dims=True)

            p = tf.reshape(p, shape=(-1, p.shape[1], p.shape[2] * p.shape[3]))
            h = tf.reshape(h, shape=(-1, h.shape[1], h.shape[2] * h.shape[3]))

            p = tf.multiply(p, p_sum)
            h = tf.multiply(h, tf.matrix_transpose(h_sum))
        else:
            p = tf.reshape(p, shape=(-1, p.shape[1], p.shape[2] * p.shape[3]))
            h = tf.reshape(h, shape=(-1, h.shape[1], h.shape[2] * h.shape[3]))

        p = tf.expand_dims(p, axis=3)
        h = tf.expand_dims(h, axis=3)

        p = tf.layers.conv2d(p,
                             filters=args.cnn2_filters,
                             kernel_size=(args.filter_width, args.cnn1_filters))
        h = tf.layers.conv2d(h,
                             filters=args.cnn2_filters,
                             kernel_size=(args.filter_width, args.cnn1_filters))

        p = self.dropout(p)
        h = self.dropout(h)

        p_all = tf.reduce_mean(p, axis=1)
        h_all = tf.reduce_mean(h, axis=1)

        x = tf.concat((p_all, h_all), axis=2)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))

        out = tf.layers.dense(x, 50)
        logits = tf.layers.dense(out, 2)

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
