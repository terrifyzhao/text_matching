import tensorflow as tf
from diin import args
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Graph:
    def __init__(self, word_embedding=None):
        self.p_c = tf.placeholder(name='p', shape=(None, args.max_char_len), dtype=tf.int32)
        self.h_c = tf.placeholder(name='h', shape=(None, args.max_char_len), dtype=tf.int32)
        self.p_w = tf.placeholder(name='p_word', shape=(None, args.max_word_len), dtype=tf.int32)
        self.h_w = tf.placeholder(name='h_word', shape=(None, args.max_word_len), dtype=tf.int32)
        self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)
        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)

        self.char_embed = tf.get_variable(name='char_embed', shape=(args.char_vocab_len, args.char_embedding_len),
                                          dtype=tf.float32)
        self.word_embed = tf.get_variable(name='word_embed', initializer=word_embedding, dtype=tf.float32,
                                          trainable=False)

        self.self_w = tf.get_variable(name='self_w', shape=(args.d * 3, 30))
        self.gate_w1 = tf.get_variable(name='gate_w1', shape=(args.d * 2, args.d))
        self.gate_w2 = tf.get_variable(name='gate_w2', shape=(args.d * 2, args.d))
        self.gate_w3 = tf.get_variable(name='gate_w3', shape=(args.d * 2, args.d))
        self.gate_b1 = tf.get_variable(name='gate_b1', shape=(args.d,))
        self.gate_b2 = tf.get_variable(name='gate_b2', shape=(args.d,))
        self.gate_b3 = tf.get_variable(name='gate_b3', shape=(args.d,))

        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, self.keep_prob)

    def encode(self, v):
        attention = tf.einsum("ijk,kl->ijl", tf.concat((v, v, tf.multiply(v, v)), axis=-1), self.self_w)
        v_hat = tf.matmul(tf.nn.softmax(attention), v)

        p_concat = tf.concat((v, v_hat), axis=-1)
        z = tf.nn.tanh(tf.einsum("ijk,kl->ijl", p_concat, self.gate_w1) + self.gate_b1)
        r = tf.nn.sigmoid(tf.einsum("ijk,kl->ijl", p_concat, self.gate_w2) + self.gate_b2)
        f = tf.nn.sigmoid(tf.einsum("ijk,kl->ijl", p_concat, self.gate_w3) + self.gate_b3)
        res = tf.multiply(r, v) + tf.multiply(f, z)
        return res

    def dense_net(self, v):
        filters = args.d * args.dense_eta
        v_in = tf.layers.conv2d(v, filters=filters, kernel_size=(1, 1))
        for _ in range(3):
            for _ in range(8):
                v_out = tf.layers.conv2d(v_in,
                                         filters=args.dense_g,
                                         kernel_size=(3, 3),
                                         padding='SAME',
                                         activation='relu')
                v_in = tf.concat((v_in, v_out), axis=-1)
            transition = tf.layers.conv2d(v_in,
                                          filters=int(v_in.shape[-1].value * args.dense_theta),
                                          kernel_size=(1, 1))
            transition_out = tf.layers.max_pooling2d(transition,
                                                     pool_size=(2, 2),
                                                     strides=2)
            v_in = transition_out
        return v_in

    def forward(self):
        p_char_embedding = tf.nn.embedding_lookup(self.char_embed, self.p_c)
        h_char_embedding = tf.nn.embedding_lookup(self.char_embed, self.h_c)

        p_word_embedding = tf.nn.embedding_lookup(self.word_embed, self.p_w)
        h_word_embedding = tf.nn.embedding_lookup(self.word_embed, self.h_w)

        p = tf.concat((p_char_embedding, p_word_embedding), axis=1)
        h = tf.concat((h_char_embedding, h_word_embedding), axis=1)

        # Encoding Layer
        with tf.variable_scope('p_encode', reuse=None):
            p_encode = self.encode(p)
        with tf.variable_scope('h_encode', reuse=None):
            h_encode = self.encode(h)
        p_encode = self.dropout(p_encode)
        h_encode = self.dropout(h_encode)

        # Interaction Layer
        I = tf.multiply(tf.expand_dims(p_encode, axis=2), tf.expand_dims(h_encode, axis=1))

        # Feature Extraction Layer
        dense_out = self.dense_net(I)
        dense_out = self.dropout(dense_out)

        # Output Layer
        dense_out = tf.reshape(dense_out, shape=(-1, dense_out.shape[1] * dense_out.shape[2] * dense_out.shape[3]))
        out = tf.layers.dense(dense_out, 256)
        out = self.dropout(out)
        self.logits = tf.layers.dense(out, args.class_size)
        self.train()

    def train(self):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        self.predict = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
