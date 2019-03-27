import torch
import torch.nn as nn

# 循环神经网络 (many-to-one)
class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        embedding_dim = args.embedding_dim
        label_num = args.label_num
        vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.bidirectional = args.bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if args.static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.lstm = nn.LSTM(embedding_dim, # x的特征维度,即embedding_dim
                            self.hidden_size,# 隐藏层单元数
                            self.layer_num,# 层数
                            batch_first=True,# 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
                            bidirectional=self.bidirectional) # 是否用双向
        self.fc = nn.Linear(self.hidden_size * 2, label_num) if self.bidirectional else nn.Linear(self.hidden_size, label_num)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(self.layer_num, x.size(0), self.hidden_size)

        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(self.layer_num, x.size(0), self.hidden_size)

        # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
        # hn,cn表示最后一个状态?维度与h0和c0一样
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 我们只需要最后一步的输出,即(batch_size, -1, output_size)
        out = self.fc(out[:, -1, :])
        return out

