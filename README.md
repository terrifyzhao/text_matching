# text_matching
文本匹配模型

本项目包含目前大部分文本匹配模型，持续更新中，其中论文解读请点击[文本相似度，文本匹配模型归纳总结](https://blog.csdn.net/u012526436/article/details/90179466)

数据集为QA_corpus，训练数据10w条，验证集和测试集均为1w条

其中对应模型文件夹下的`args.py`文件是超参数

训练：
`python train.py`

测试：
`python test.py`

测试集结果对比：

模型 | loss | acc | 论文地址
:-: | :-: | :-: | :-: |
DSSM | 0.7613157 | 0.6864 | [DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) |
ConvNet | 0.6872447 | 0.6977 | [ConvNet](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf) |
ESIM | 0.55444807| 0.736 | [ESIM](https://arxiv.org/pdf/1609.06038.pdf) |
ABCNN | 0.5771452| 0.7503 | [ABCNN](https://arxiv.org/pdf/1512.05193.pdf) |
BiMPM | 0.4852| 0.764 | [BiMPM](https://arxiv.org/pdf/1702.03814.pdf) |
DIIN | 0.48298636| 0.7694 | [DIIN](https://arxiv.org/pdf/1709.04348.pdf) |

以上测试结果可能不是模型的最优解，超参的选择也不一定是最优的，如果你想用到自己的实际工程中，请自行调整超参