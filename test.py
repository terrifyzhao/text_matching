import pandas as pd
from esim.graph import Graph
import tensorflow as tf
from utils.load_data import char_index
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# data = pd.read_csv('ccb/data_5_7.csv')
data = pd.read_csv('ccb/sim.txt', sep='\t')
sens = []
indexs = []
with open('ccb/test_5_7.txt', encoding='utf-8')as file:
    for line in file.readlines():
        sen, index = line.split('\t')
        sens.append(sen)
        indexs.append(str(index).strip())

model = Graph()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, 'output/esim/esim_5.ckpt')

data_id = data['id'].values
data_sen = data['sentence'].values


def predict(p, h):
    prob, prediction = sess.run([model.prob, model.prediction],
                                feed_dict={model.p: p,
                                           model.h: h,
                                           model.keep_prob: 1})

    # pos = [i[1] for i in prob]
    # p = pos.index(max(pos))
    # print('prob:', max(pos))
    # print(str(prob))
    # return p
    return prob


def test():
    for value, index in zip(sens, indexs):
        sen = [value] * len(data)
        s1, s2 = char_index(sen, data_sen)
        predict_index = predict(s1, s2)
        d = data.values[predict_index]
        # t = test_data.values[index]
        print(d, ' - ', index)
        print(data.index[predict_index], ' - ', value)
        print('\n')


def test2():
    for value, index in zip(sens, indexs):
        sen = [value] * len(data)
        s1, s2 = char_index(sen, data_sen)
        probs = []
        for i, j in zip(s1, s2):
            prob = predict([i], [j])
            probs.append(prob[0][1])

        print(len(probs), probs)
        predict_index = probs.index(max(probs))
        print(max(probs))

        print(data_id[predict_index], ' - ', index)
        print(data_sen[predict_index], ' - ', value)
        print('\n')


def test3():
    while True:
        sen1 = input('sen1:')
        sen2 = input('sen2:')
        s1, s2 = char_index([sen1], [sen2])
        print(predict(s1, s2))


if __name__ == '__main__':
    test2()
