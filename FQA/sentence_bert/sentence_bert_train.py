# encoding:utf-8
"""
# @Project -> File :QA -> sentence_bert_train.py
# @IDE    : PyCharm
# @Author : eason wong
# @Time   : 2023/5/14 0014 20:40
# @Desc   : 
"""
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, losses
import logging
import os
import csv

# 读取训练文本数据
train_data_path = '/data'
sentences = []
with os.scandir(train_data_path) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.endswith('.txt'):
            with open(entry.path, 'r', encoding='utf-8') as f:
                text = f.read()
                # 将文本分割成句子
                sentences += [s.strip() for s in text.split('.') if s.strip()]

# 初始化Bert模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# 构建SentencesDataset，将文本转化为Bert的输入格式
train_data = SentencesDataset(sentences, tokenizer, max_seq_length=128)

# 初始化SentenceTransformer模型
sentence_bert_model = SentenceTransformer('bert-base-uncased')

# 定义训练的loss函数和优化器
train_loss = losses.CosineSimilarityLoss(model=sentence_bert_model)
train_optimizer = sentence_bert_model.get_optimizer(1e-5, 1e-5)

# 开始训练
train_dataloader = sentence_bert_model.get_batch_loader(train_data, batch_size=16, shuffle=True)
sentence_bert_model.fit(train_objectives=[(train_dataloader, train_loss)],
                        epochs=3,
                        warmup_steps=100,
                        optimizer=train_optimizer,
                        output_path='/model',
                        save_best_model=True)

# 保存模型
sentence_bert_model.save('mymodel.bin')