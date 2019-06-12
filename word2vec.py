# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import math
import os
import random
import sys
import pandas as pd
import jieba

import numpy as np
import tensorflow as tf
import re

from tensorflow.contrib.tensorboard.plugins import projector

data_index = 0
data_num = 0


def word2vec_basic(log_dir):
    def load_data():

        df = pd.read_csv('input/origin_5_16.csv', encoding='utf-8-sig')
        df2 = pd.read_csv('input/query_5_16.csv', encoding='utf-8-sig')
        df_sentences = np.concatenate([df['sentence'].values, df2['sentence'].values])
        segments = list(
            map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), df_sentences))

        return segments

    def build_dataset(words, min_count):
        """Process raw inputs into a dataset."""
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common())
        for i, v in enumerate(count[::-1]):
            if v[1] > min_count:
                count = count[0:len(count) - i].copy()
                break

        dictionary = {}
        for word, _ in count:
            if word != '\u200d':
                dictionary[word] = len(dictionary)

        data = []
        sentence_data = []
        unk_count = 0
        for word in words:
            if word == 'PAD':
                data.append(sentence_data.copy())
                sentence_data.clear()
                continue
            index = dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            sentence_data.append(index)

        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary

    vocabulary = []
    for sentence in load_data():
        tmp = sentence.copy()
        tmp.append('PAD')
        vocabulary.extend(tmp)

    all_data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, 0)
    vocabulary_size = len(dictionary)
    del vocabulary  # Hint to reduce memory.
    print('Most common words', count[2:7])

    # print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # Step 3: Function to generate a training batch for the skip-gram model.
    # num_skips表示的是选择几对数据，例如skip_window如果是2，那么就是有4个词和target相关，如果num_skips
    # 是3，就随机选择三个词与target组队进行训练
    # skip_window是单边的window大小，如果是2，那总的window大小就是5
    def generate_batch(batch_size, num_skips, skip_window):
        global data_index
        global data_num
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)  # 输入
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  # 输出
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]

        if data_num >= len(all_data):
            data_num = 0
        data = all_data[data_num]
        # 建立一个队列
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
        # 超出最大数据了
        while data_index + span > len(data):
            data_num += 1
            data = all_data[data_num]
        # 把前三个值放到buffer里
        buffer.extend(data[data_index:data_index + span])
        # 每次增加一个span
        data_index += span
        # 要循环多少次才能把batch填满，每次填num_skips个数
        for i in range(batch_size // num_skips):
            # 确认context的位置index
            context_words = [w for w in range(span) if w != skip_window]
            # 随机选择num_skips个位置，并返回indexjiu fenlie
            words_to_use = random.sample(context_words, num_skips)
            # 遍历context的index并与target组合在一起
            for j, context_word in enumerate(words_to_use):
                # batch保存的是输入数据，因为是skip-gram，所以把target放进去，
                batch[i * num_skips + j] = buffer[skip_window]
                # labels保存输出，根据words_to_use中的下标context_word去获取context word
                labels[i * num_skips + j, 0] = buffer[context_word]  # 输出
            # 如果遍历完了整个data则把buffer重新添加data首的span个元素，data_index重置
            # 否则buffer添加一个元素，队列中首的数据挤掉，尾部加入新数据，index+1，
            if data_index == len(data):
                data_num += 1
                if data_num >= len(all_data):
                    data_num = 0
                data = all_data[data_num]
                data_index = 0
                while data_index + span > len(data):
                    data_num += 1
                    data = all_data[data_num]
                data_index = span
                buffer.extend(data[0:0 + span])
            else:
                buffer.append(data[data_index])
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        # data_index = (data_index + len(data) - span) % len(data)
        data_index = data_index - span
        return batch, labels

    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
              reverse_dictionary[labels[i, 0]])

    # Step 4: Build and train a skip-gram model.

    batch_size = 256
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit
    # the validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        with tf.name_scope('inputs'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(
                    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size))

        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all
        # embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Merge all summaries.
        merged = tf.summary.merge_all()

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a saver.
        saver = tf.train.Saver()

    # Step 5: Begin training.
    num_steps = 100001

    with tf.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(log_dir, session.graph)

        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                        skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned
            # "summary" variable. Feed metadata variable to session for visualizing
            # the graph in TensorBoard.
            _, summary, loss_val, embedding = session.run([optimizer, merged, loss, embeddings],
                                                          feed_dict=feed_dict,
                                                          run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000
                # batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

        # Write corresponding labels for the embeddings.
        with open(log_dir + '/word_vocab.tsv', 'w', encoding='utf-8') as f:
            for i in range(vocabulary_size):
                f.write(reverse_dictionary[i] + '\n')

        # Save the model for checkpoints.
        saver.save(session, os.path.join(log_dir, 'model.ckpt'))

        with open(log_dir + '/w2v.vec', 'wb')as file:
            import pickle
            pickle.dump(embedding, file)
            print('save w2v done')

        # Create a configuration for visualizing embeddings with the labels in
        # TensorBoard.
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

    writer.close()

    # Step 6: Visualize the embeddings.

    # pylint: disable=missing-docstring
    # Function to draw visualization of distance between embeddings.
    def plot_with_labels(low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

        plt.savefig(filename)

    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join(log_dir,
                                                            'tsne.png'))

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)


# All functionality is run after tf.compat.v1.app.run() (b/122547914). This
# could be split up but the methods are laid sequentially with their usage for
# clarity.
def main(unused_argv):
    # Give a folder path as an argument with '--log_dir' to save
    # TensorBoard summaries. Default is a log folder in current directory.
    word2vec_basic('output/word2vec')


if __name__ == '__main__':
    tf.app.run()
