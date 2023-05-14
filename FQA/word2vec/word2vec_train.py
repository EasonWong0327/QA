# encoding:utf-8
"""
# @Project -> File :QA -> word2vec_train.py
# @IDE    : PyCharm
# @Author : eason wong
# @Time   : 2023/5/14 0014 17:23
# @Desc   : 
"""
import gensim
import os


class DataLoad(object):
    '''
    训练过程中数据会很大，使用迭代器
    '''
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for file in os.listdir(self.filename):
            for line in open(os.path.join(self.filename, file)):
                yield line.split()

# 定义数据集路径
sentences = DataLoad('/all_txt')

# 配置训练参数
model = gensim.models.Word2Vec(sentences,min_count=5, size=100, workers=4, window=5)

# 保存模型
model.save('vertical_word2vec.model')



# 使用模型
from gensim.models import KeyedVectors

# 加载已经训练好的word2vec模型
model = KeyedVectors.load('vertical_word2vec.model')

# 获取单词的词向量
vector = model['word']