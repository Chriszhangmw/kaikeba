
import numpy as np
import torch
from torch import nn, optim
import random
from collections import Counter
import matplotlib.pyplot as plt

#训练数据
text = "The tool counts the number of times each combination of two words " \
       "appears in the training text, " \
       "and then these counts are used in an equation to determine which word combinations to turn into phrases. " \
       "The equation is designed to make phrases out of words which occur together often relative to the number o" \
       "f individual occurrences. It also favors phrases made of infrequent words in order to avoid making phrases " \
       "out of common words like"

#参数设置
EMBEDDING_DIM = 2 #词向量维度
PRINT_EVERY = 1000 #可视化频率
EPOCHS = 1000 #训练的轮数
BATCH_SIZE = 15 #每一批训练数据大小
N_SAMPLES = 3 #负样本大小
WINDOW_SIZE = 5 #周边词窗口大小, BATCH_SIZE一定要大于WINDOW_SIZE。

FREQ = 0 #词汇出现频率
DELETE_WORDS = False #是否删除部分高频词

#文本预处理
def prepocess(text,FREQ):
    text = text.lower()
    words = text.split()
    #去除低频词
    word_couts = Counter(words)
    trimmed_words = [word for word in words if word_couts[word] > FREQ]
    return  trimmed_words

words = prepocess(text,FREQ)

#构建词典
vocab = set(words)
vocab2int = {w:c for c,w in enumerate(vocab)}
int2vocab = {c:w for c,w in enumerate(vocab)}


#将文本转化为数值
int_words = [vocab2int[w] for w in words]


#计算单词频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
word_freqs = {w:c/total_count for w,c in int_word_counts.items()}


#去除出现频次高的词汇
if DELETE_WORDS:
    t = 1e-5
    prob_drop = {w:1-np.sqrt(t/word_freqs[w]) for w in int_word_counts}
    train_words = [w for w in int_words if random.random() < (1 - prob_drop[w])]
else:
    train_words = int_words


#单词分布
word_freqs = np.array(list(word_freqs.values()))
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))



#获取目标词汇
#The tool counts the number of times each combination of two words
def get_target(words,idx, WINDOW_SIZE):
    target_window = np.random.randint(1,WINDOW_SIZE + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point:idx] + words[idx+1:end_point+1])
    return list(targets)


#批次化数据
def get_batch(words,batch_size,windows_size):
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]
    for idx in range(0,len(words),batch_size):
        batch_x,batch_y = [],[]
        batch = words[idx:idx+batch_size]#The tool counts the number of times each combination of two words
        for i in range(len(batch)):
            x = batch[i]
            y = get_target(batch,i,windows_size)
            batch_x.extend([x] * len(y))
            batch_y.extend(y)
        yield batch_x, batch_y

#定义模型
class SkipGramNeg(nn.Module):
    def __init__(self,n_vocab,n_emded,noise_dist):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_emded = n_emded
        self.noise_dist = noise_dist
        #定义词向量
        self.in_emded = nn.Embedding(n_vocab,n_emded)
        # [The tool counts the number of]
        # [2,   3,    1,    2,  10,  19]
        self.out_embed = nn.Embedding(n_vocab,n_emded)

        #初始化
        self.in_emded.weight.data.uniform_(-1,1)
        self.out_embed.weight.data.uniform_(-1, 1)
    #forward过程
    def forward_inputs(self,input_words):
        input_vectors = self.in_emded(input_words)
        return input_vectors

    def forward_output(self,output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self,size,N_samples):# [The tool counts the number of]
        noise_dist = self.noise_dist# [2,3]  [2,3].T= [3,2],reshape
        noise_words = torch.multinomial(noise_dist,size * N_samples,replacement=True)
        noise_vector = self.out_embed(noise_words).view(size,N_samples,self.n_emded)
        return noise_vector

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input_vectors,output_vectors,noise_vectors):
        batch_size,embed_size = input_vectors.shape #(batch_size,embed_size)
        #维度转换
        input_vectors = input_vectors.view(batch_size,embed_size,1)
        output_vectors = output_vectors.view(batch_size,1,embed_size)

        #正样本的损失
        out_loss = torch.bmm(output_vectors,input_vectors).sigmoid().log()
        #out_loss shape is [batch_size,1,1] [1,n] * [n,1] = [1,1]
        out_loss = out_loss.squeeze()
        #负样本损失
        noise_loss = torch.bmm(noise_vectors.neg(),input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)

        return -(out_loss + noise_loss).mean()

model = SkipGramNeg(len(vocab2int),EMBEDDING_DIM,noise_dist=noise_dist)
criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

steps = 0
for e in range(EPOCHS):
    for input_words,target_words in get_batch(train_words,BATCH_SIZE,WINDOW_SIZE):
        steps +=1
        inputs,targets = torch.LongTensor(input_words),torch.LongTensor(target_words)
        input_vectors = model.forward_inputs(inputs)
        output_vectors = model.forward_output(targets)
        size,_ = input_vectors.shape
        noise_vectors = model.forward_noise(size,N_SAMPLES)
        loss = criterion(input_vectors,output_vectors,noise_vectors)
        if steps%PRINT_EVERY == 0:
            print("loss :",loss)
        #backwords
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for i, w in int2vocab.items():
    vectors = model.state_dict()["in_embed.weight"]
    x,y = float(vectors[i][0]),float(vectors[i][1])
    plt.scatter(x,y)
    plt.annotate(w, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()