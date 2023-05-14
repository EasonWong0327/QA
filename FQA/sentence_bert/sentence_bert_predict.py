# encoding:utf-8
"""
# @Project -> File :QA -> sentence_bert_predict.py
# @IDE    : PyCharm
# @Author : eason wong
# @Time   : 2023/5/14 0014 20:42
# @Desc   : 
"""
from sentence_transformers import SentenceTransformer
import torch

# 加载自定义的sentence-bert模型
model = SentenceTransformer('mymodel.bin')

# 两个句子的文本
sentence1 = '今天吃点什么？'
sentence2 = '今晚去吃啥好呢'

# 计算相似度
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)

print('句子1：', sentence1)
print('句子2：', sentence2)
print('相似度：', similarity.item())
