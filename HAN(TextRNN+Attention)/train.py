import argparse
import os
import sys
import torch
import torch.nn.functional as F
import data_processor
from model import AttentionWordGRU, AttentionSentGRU
parser = argparse.ArgumentParser(description='TextRNN text classifier')

parser.add_argument('-lr', type=float, default=0.01, help='学习率')
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=20)
parser.add_argument('-embedding-dim', type=int, default=128, help='词向量的维度')
parser.add_argument('-word-hidden-size', type=int, default=64, help='word层中GRU神经单元数')
parser.add_argument('-sent-gru-hidden-size', type=int, default=64, help='sent层中GRU神经单元数')
parser.add_argument('-layer-num', type=int, default=1, help='gru stack的层数')
parser.add_argument('-label-num', type=int, default=2, help='标签个数')
parser.add_argument('-bidirectional', type=bool, default=False, help='是否使用双向lstm')
parser.add_argument('-static', type=bool, default=False, help='是否使用预训练词向量')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=20, help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')
parser.add_argument('-vocab-size', type=int, default='50000', help='词典大小,会根据训练集和验证集计算出来,此处占位')
parser.add_argument('-iterations', type=int, default=400, help='每个epoch迭代次数,会根据batch_size和训练集数量计算出来,此处占位')

args = parser.parse_args()

def train(args):
    text = data_processor.load_data(args)
    train_iter = data_processor.gen_batch(args, text)
    dev_iter = data_processor.gen_test(text, args)
    print('加载数据完成')
    word_attn_model = AttentionWordGRU(args)
    sent_attn_model = AttentionSentGRU(args)
    model = None
    if args.cuda: model.cuda()
    word_optimizer = torch.optim.Adam(word_attn_model.parameters(), lr=args.lr)
    sent_optimizer = torch.optim.Adam(sent_attn_model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epoch + 1):
        for i in range(args.iterations):
            word_optimizer.zero_grad()
            sent_optimizer.zero_grad()
            doc_texts, targets = next(train_iter)

            # doc_texts的维度为(batch_size, sents_num, words_num)
            word_attn_vectors = None
            for doc_text in doc_texts:
                # word_attn_vector的维度为(sent_num, hidden_size)
                word_attn_vector = word_attn_model(doc_text)
                # 将word_attn_vector的维度变为(1, sent_num, hidden_size)
                word_attn_vector = word_attn_vector.unsqueeze(0)
                if word_attn_vectors is None:
                    word_attn_vectors = word_attn_vector
                else:
                    # word_attn_vectors的维度为(batch_size, sent_num, hidden_size)
                    word_attn_vectors = torch.cat((word_attn_vectors, word_attn_vector), 0)
            logits = sent_attn_model(word_attn_vectors)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            word_optimizer.step()
            sent_optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                # torch.max(logits, 1)函数：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                corrects = (torch.max(logits, 1)[1] == targets).sum()
                train_acc = 100.0 * corrects / args.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             args.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, word_attn_model, sent_attn_model)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(word_attn_model, args.save_dir, 'best', steps)
                        save(sent_attn_model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt

'''
对验证集进行测试 
由于对doc和sentence进行pad需要大量的计算量,把所有验证集pad成相同大小计算量过大,因此这里简单的用一个batch_size的验证集进行验证
'''
def eval(data_iter, word_attn_model, sent_attn_model):
    corrects, avg_loss = 0, 0
    for doc_texts, targets in data_iter:

        # doc_texts的维度为(batch_size, sents_num, words_num)
        word_attn_vectors = None
        for doc_text in doc_texts:
            # word_attn_vector的维度为(sent_num, hidden_size)
            word_attn_vector = word_attn_model(doc_text)
            # 将word_attn_vector的维度变为(1, sent_num, hidden_size)
            word_attn_vector = word_attn_vector.unsqueeze(0)
            if word_attn_vectors is None:
                word_attn_vectors = word_attn_vector
            else:
                word_attn_vectors = torch.cat((word_attn_vectors, word_attn_vector), 0)
        logits = sent_attn_model(word_attn_vectors)
        loss = F.cross_entropy(logits, targets)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(targets.size()) == targets).sum()
        # 用一个batch_size的验证集进行验证
        break

    accuracy = 100.0 * corrects / args.batch_size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       args.batch_size))
    return accuracy

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

train(args)
