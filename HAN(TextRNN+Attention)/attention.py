import torch

def batch_matmul(seq, weight, activation=''):
    # seq维度为(batch_size, seq_length, hidden_size)
    # weight此时为query_vec,维度为(hidden_size, 1)
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if (activation == 'tanh'): s = torch.tanh(_s)

        # 将_s的维度从(seq_length, 1)变为(1, seq_length, 1)
        _s = _s.unsqueeze(0)

        if (s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    # 经运算，s维度为(batch_size, seq_length, 1),经squeeze变为(batch_size, seq_length)
    return s.squeeze()


def batch_matmul_bias(seq, weight, bias, activation=''):
    # seq维度为(batch_size, seq_length, hidden_size)
    # weight维度为(hidden_size, hidden_size)
    # bias维度为(hidden_size, 1)
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)  # _s维度为(seq_length, hidden_size)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).t()  # _s_bias维度为(seq_length, hidden_size)
        if (activation == 'tanh'):
            _s_bias = torch.tanh(_s_bias)
        # 将_s_bias的维度从(seq_length, hidden_size)变为(1, seq_length, hidden_size)
        _s_bias = _s_bias.unsqueeze(0)
        if (s is None):
            s = _s_bias
        else:
            s = torch.cat((s, _s_bias), 0)
    # s的维度为(batch_size, seq_length, hidden_size)
    return s


def attention_mul(rnn_outputs, att_weights):
    # rnn_outputs的维度为(batch_size, seq_length, hidden_size)
    # att_weights的维度为(batch_size, seq_length)
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]  # h_i维度为(seq_length, hidden_size)
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)  # a_i维度为(seq_length, hidden_size)
        h_i = a_i * h_i  # 运算后的h_i维度为(seq_length, hidden_size)
        h_i = h_i.unsqueeze(0)  # h_i的维度为(1, seq_length, hidden_size)
        if (attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    # attn_vetors维度为(batch_size, seq_length, hidden_size)
    # 经过sum, attn_vectors维度为(batch_size, hidden_size)
    return torch.sum(attn_vectors, 1)