# text_matching
文本匹配模型

本项目包含目前大部分文本匹配模型，持续更新中，其中论文解读请点击[文本相似度，文本匹配模型归纳总结](https://blog.csdn.net/u012526436/article/details/90179466)

数据集为QA_corpus，训练数据10w条，验证集和测试集均为1w条

其中对应模型文件夹下的`args.py`文件是超参数

训练：
`python train.py`

测试：
`python test.py`

词向量：
不同的模型输入不一样，有的模型的输入只有简单的字向量，有的模型换成了字向量+词向量，甚至还有静态词向量(训练过程中不进行更新)和
动态词向量(训练过程中更新词向量)，所有不同形式的输入均以封装好，调用方法如下


静态词向量，请执行
`python word2vec_gensim.py`，该版本是采用gensim来训练词向量

动态词向量，请执行
`python word2vec.py`，该版本是采用tensorflow来训练词向量，训练完成后会保存embedding矩阵、词典和词向量在二维矩阵的相对位置的图片，
如果非win10环境，由于字体的原因图片可能保存失败

测试集结果对比：

模型 | loss | acc | 输入说明 | 论文地址
:-: | :-: | :-: | :-: | :-: |
DSSM | 0.7613157 | 0.6864 | 字向量 | [DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) |
ConvNet | 0.6872447 | 0.6977 | 字向量 | [ConvNet](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf) |
ESIM | 0.55444807| 0.736 | 字向量 | [ESIM](https://arxiv.org/pdf/1609.06038.pdf) |
ABCNN | 0.5771452| 0.7503 | 字向量 | [ABCNN](https://arxiv.org/pdf/1512.05193.pdf) |
BiMPM | 0.4852| 0.764 | 字向量+静态词向量 | [BiMPM](https://arxiv.org/pdf/1702.03814.pdf) |
DIIN | 0.48298636| 0.7694 | 字向量+动态词向量 | [DIIN](https://arxiv.org/pdf/1709.04348.pdf) |
DRCN | 0.6549849 | 0.7811 | 字向量+静态词向量+动态词向量+是否有相同词 | [DRCN](https://arxiv.org/pdf/1805.11360.pdf) |

以上测试结果可能不是模型的最优解，超参的选择也不一定是最优的，如果你想用到自己的实际工程中，请自行调整超参
