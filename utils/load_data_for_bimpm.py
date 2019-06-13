import os
import pandas as pd
from bimpm import args
import numpy as np
from gensim.models import Word2Vec
import jieba
from utils.data_utils import pad_sequences, shuffle, one_hot
import re

model = Word2Vec.load('../output/bimpm/word2vec.model')


def load_char_vocab():
    path = os.path.join(os.path.dirname(__file__), '../input/vocab.txt')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(p_sentences, h_sentences):
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.max_char_len)
    h_list = pad_sequences(h_list, maxlen=args.max_char_len)

    return p_list, h_list


def w2v(word, dynamic=True):
    if dynamic:
        vocab = []
        # with open('../out/word2vec/w2v.vec')as file:
        #     import pickle
        #     embedding = pickle.load(file)
        with open('../output/word2vec/word_vocab.tsv', encoding='utf-8')as file:
            for line in file.readlines():
                vocab.append(line.strip())

        # one = one_hot(vocab.index(word), len(vocab))
        # one = []
        index = []
        for w in word:
            if len(w.strip()) > 0 and w != '\u200d':
                index.append(vocab.index(w))

        return one_hot(index, len(vocab))
    else:
        return model.wv[word]


def w2v_process(vec):
    if len(vec) > args.max_word_len:
        vec = vec[0:args.max_word_len]
    elif len(vec) < args.max_word_len:
        zero = np.zeros(args.word_embedding_len)
        length = args.max_word_len - len(vec)
        for i in range(length):
            vec = np.vstack((vec, zero))
    return vec


def load_data(file, data_size=None, dynamic=True):
    path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_index, h_index = create_data(p, h)

    p_seg = map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), p)
    h_seg = map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), h)
    # p_seg = map(lambda x: list(jieba.cut(x)), p)
    # h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_vec = list(map(lambda x: w2v(x, dynamic), p_seg))
    h_vec = list(map(lambda x: w2v(x, dynamic), h_seg))

    # if not dynamic:
    # p_vec = np.array(list(map(lambda x: w2v_process(x), p_vec)))
    # h_vec = np.array(list(map(lambda x: w2v_process(x), h_vec)))

    return p_index, h_index, p_vec, h_vec, label


def load_fake_data():
    p, h, label = [], [], []
    for i in range(10):
        p.append(np.arange(10))
        h.append(np.arange(10))
        label.append(1)
    return p, h, label


if __name__ == '__main__':
    load_data('../input/dev.csv')
