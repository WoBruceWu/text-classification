import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import attention

# 词语级GRU
class AttentionWordGRU(nn.Module):
    def __init__(self, args):
        super(AttentionWordGRU, self).__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.word_gru_hidden_size = args.word_hidden_size
        self.bidirectional = args.bidirectional

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if args.static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.direction_num = 2 if self.bidirectional else 1

        # 词语对应的GRU层
        self.word_gru = nn.GRU(self.embedding_dim, self.word_gru_hidden_size,
                               bidirectional=self.bidirectional, batch_first=True)

        # attention参数中矩阵参数，维度为(hidden_size*direction_num, hidden_size*direction_num)
        self.weights_w_word = nn.Parameter(torch.Tensor(self.word_gru_hidden_size*self.direction_num, self.word_gru_hidden_size*self.direction_num))

        # attention参数中矩阵对应的偏差项，维度为(hidden_size*direction_num, 1)
        self.bias_word = nn.Parameter(torch.Tensor(self.word_gru_hidden_size*self.direction_num, 1))

        # 对每个词的表示做attention的向量
        self.query_vec_word = nn.Parameter(torch.Tensor(self.word_gru_hidden_size*self.direction_num, 1))

        # 初始化attention矩阵和向量
        self.weights_w_word.data.uniform_(-0.1, 0.1)
        self.query_vec_word.data.uniform_(-0.1, 0.1)



    def forward(self, x):
        # 输入x的维度为(sent_num, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(x)  # 经过embedding,x的维度为(sent_num, time_step, input_size=embedding_dim)

        # GRU隐层初始化,维度为(num_layers*direction_num, sent_num, hidden_size)，本论文结构中
        h0 = torch.zeros(self.direction_num, x.size(0), self.word_gru_hidden_size)

        # GRU层运算,out的维度为(sent_num, seq_length, hidden_size)
        out, hn = self.word_gru(x, h0)

        # word_squish维度为(sent_num, seq_length, hidden_size)
        word_squish = attention.batch_matmul_bias(out, self.weights_w_word, self.bias_word, 'tanh')

        # word_attn维度为(sent_num, seq_length)
        word_attn = attention.batch_matmul(word_squish, self.query_vec_word, '')

        # word_attn_norm维度为(sent_num, seq_length)
        word_attn_norm = F.softmax(word_attn)


        word_attn_vectors = attention.attention_mul(out, word_attn_norm)
        # word_attn_vectors:(sent_num, hidden_size)
        # hn: (num_layers*direction_num, sent_num, hidden_size)
        # word_attn_norm:(sent_num, seq_length)
        return word_attn_vectors



# ## 句子级attention

class AttentionSentGRU(nn.Module):

    def __init__(self, args):

        super(AttentionSentGRU, self).__init__()
        label_num = args.label_num
        word_gru_hidden_size = args.word_hidden_size
        sent_gru_hidden_size = args.sent_gru_hidden_size
        bidirectional = args.bidirectional

        direction_num = 2 if bidirectional else 1
        self.sent_gru_hidden_size = sent_gru_hidden_size
        self.direction_num = direction_num

        # 句子对应的GRU层
        self.sent_gru = nn.GRU(direction_num*word_gru_hidden_size, sent_gru_hidden_size,
                               bidirectional=bidirectional, batch_first=True)

        # attention参数中矩阵参数，维度为(hidden_size*direction_num, hidden_size*direction_num)
        self.weight_w_sent = nn.Parameter(torch.Tensor(direction_num * sent_gru_hidden_size, direction_num * sent_gru_hidden_size))

        # attention参数中矩阵对应的偏差项，维度为(hidden_size*direction_num, 1)
        self.bias_sent = nn.Parameter(torch.Tensor(direction_num * sent_gru_hidden_size, 1))

        # 对每个句子的表示做attention的向量
        self.query_vec_sent = nn.Parameter(torch.Tensor(direction_num * sent_gru_hidden_size, 1))

        self.linear = nn.Linear(sent_gru_hidden_size, label_num)

        # 初始化attention矩阵和向量
        self.weight_w_sent.data.uniform_(-0.1, 0.1)
        self.query_vec_sent.data.uniform_(-0.1, 0.1)

    # word_attn_vectors的维度为(batch_size, sent_num, word_hidden_size),sent_num为一篇文章中句子的数量,batch_size为文章的数量
    def forward(self, word_attn_vectors):

        # h0维度为((num_layers*direction_num, batch_size, sent_hidden_size))
        h0 = torch.zeros(self.direction_num, word_attn_vectors.size(0), self.sent_gru_hidden_size)

        # GRU层运算,out的维度为(batch_size, sent_length, sent_hidden_size)
        out, hn = self.sent_gru(word_attn_vectors, h0)

        # sent_squish的维度为(batch_size, sent_length, sent_hidden_size)
        sent_squish = attention.batch_matmul_bias(out, self.weight_w_sent, self.bias_sent, activation='tanh')

        # sent_attn的维度为(batch_size, sent_length)
        sent_attn = attention.batch_matmul(sent_squish, self.query_vec_sent)

        # sent_attn的维度为(batch_size, sent_length)
        sent_attn_norm = F.softmax(sent_attn)

        # sent_attn_vectors的维度为(batch_size, sent_hidden_size)
        sent_attn_vectors = attention.attention_mul(out, sent_attn_norm)

        # 全连接层，返回的logits维度为(batch_size, label_num)
        logits = self.linear(sent_attn_vectors)

        return logits
