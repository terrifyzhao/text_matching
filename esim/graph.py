import tensorflow as tf
from esim import args


class Graph:

    def __init__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=(100, args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(100, args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=100, name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')

        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
                                         name='embedding')

        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def bilstm(self, x, hidden_size):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, activation='tanh')
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, activation='tanh')

        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    @staticmethod
    def cosine(p, h):
        # p_norm = tf.norm(p, axis=2, keepdims=True, ord=1)
        # h_norm = tf.norm(p, axis=2, keepdims=True, ord=1)

        cosine = tf.multiply(p, h) / tf.abs(p) * tf.abs(h)

        return cosine

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        with tf.variable_scope("lstm_p", reuse=tf.AUTO_REUSE):
            (p_f, p_b), _ = self.bilstm(p_embedding, args.embedding_hidden_size)
        with tf.variable_scope("lstm_p", reuse=tf.AUTO_REUSE):
            (h_f, h_b), _ = self.bilstm(h_embedding, args.embedding_hidden_size)

        p = tf.concat([p_f, p_b], axis=2)
        h = tf.concat([h_f, h_b], axis=2)

        a = self.dropout(p)
        b = self.dropout(h)

        e = self.cosine(p, h)

        a_attention = tf.reduce_sum(tf.matmul(e, tf.transpose(a, perm=[0, 2, 1]))) / tf.reduce_sum(e, axis=2,
                                                                                                   keepdims=True)
        b_attention = tf.reduce_sum(tf.matmul(e, tf.transpose(b, perm=[0, 2, 1]))) / tf.reduce_sum(e, axis=2,
                                                                                                   keepdims=True)

        m_a = tf.concat((a, a_attention, a - a_attention, a * a_attention), axis=2)
        m_b = tf.concat((b, b_attention, b - b_attention, b * b_attention), axis=2)

        with tf.variable_scope("lstm_a", reuse=tf.AUTO_REUSE):
            (a_f, a_b), _ = self.bilstm(m_a, args.context_hidden_size)
        with tf.variable_scope("lstm_b", reuse=tf.AUTO_REUSE):
            (b_f, b_b), _ = self.bilstm(m_b, args.context_hidden_size)

        a = tf.concat((a_f, a_b), axis=2)
        b = tf.concat((b_f, b_b), axis=2)

        a = self.dropout(a)
        b = self.dropout(b)

        a_avg = tf.reduce_mean(a, axis=2)
        b_avg = tf.reduce_mean(b, axis=2)

        a_max = tf.reduce_max(a, axis=2)
        b_max = tf.reduce_max(b, axis=2)

        # a_avg = tf.nn.avg_pool(a, ksize=(1, 512), strides=[1, 1, 1, 1], padding='VALID')
        # b_avg = tf.nn.avg_pool(b, ksize=(512, 1), strides=[1, 1, 1, 1], padding='VALID')
        #
        # a_max = tf.nn.max_pool(a, ksize=(512, 1), strides=[1, 1, 1, 1], padding='VALID')
        # b_max = tf.nn.max_pool(b, ksize=(512, 1), strides=[1, 1, 1, 1], padding='VALID')

        v = tf.concat((a_avg, a_max, b_avg, b_max), axis=1)
        v = tf.layers.dense(v, 512, activation='tanh')
        v = self.dropout(v)
        logits = tf.layers.dense(v, 2, activation='tanh')

        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
