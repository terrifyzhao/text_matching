import tensorflow as tf
from esim import args


class Graph:

    def __init__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')

        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
                                         name='embedding')

        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def bilstm(self, x, hidden_size):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        with tf.variable_scope("lstm_p", reuse=tf.AUTO_REUSE):
            (p_f, p_b), _ = self.bilstm(p_embedding, args.embedding_hidden_size)
        with tf.variable_scope("lstm_p", reuse=tf.AUTO_REUSE):
            (h_f, h_b), _ = self.bilstm(h_embedding, args.embedding_hidden_size)

        p = tf.concat([p_f, p_b], axis=2)
        h = tf.concat([h_f, h_b], axis=2)

        p = self.dropout(p)
        h = self.dropout(h)

        e = tf.matmul(p, tf.transpose(h, perm=[0, 2, 1]))
        a_attention = tf.nn.softmax(e)
        b_attention = tf.transpose(tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1])), perm=[0, 2, 1])

        a = tf.matmul(a_attention, h)
        b = tf.matmul(b_attention, p)

        m_a = tf.concat((a, p, a - p, tf.multiply(a, p)), axis=2)
        m_b = tf.concat((b, h, b - h, tf.multiply(b, h)), axis=2)

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

        v = tf.concat((a_avg, a_max, b_avg, b_max), axis=1)
        v = tf.layers.dense(v, 512, activation='tanh')
        v = self.dropout(v)
        logits = tf.layers.dense(v, 2, activation='tanh')
        self.prob = tf.nn.softmax(logits)
        self.prediction = tf.argmax(logits, axis=1)
        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
