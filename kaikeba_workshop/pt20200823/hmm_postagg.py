#初始化字典
tag2id, id2tag = {},{}
word2id,id2word = {},{}

#建立字典
for line in open('./data/traindata.txt'):
    items = line.split('/')
    word = items[0]
    tag = items[1].rstrip()
    if word not in word2id:
        word2id[word] = len(word2id)
        id2word[len(word2id)] = word
    if tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[len(tag2id)] = tag

# print(word2id)
# print(tag2id)

M = len(word2id)
N = len(tag2id)

import numpy as np

P = np.zeros(N)#初始状态矩阵
B = np.zeros((N,M)) #发射状态矩阵
A = np.zeros((N,N)) #状态转移矩阵

prev_tag = ""
for line in open('./data/traindata.txt'):
    items = line.split('/')
    wordId,tagId = word2id[items[0]],tag2id[items[1].rstrip()]
    if prev_tag == "":
        P[tagId] += 1
        B[tagId][wordId] += 1
    else:
        B[tagId][wordId] += 1
        A[tag2id[prev_tag]][tagId] += 1

    if items[0] == '.':
        prev_tag = ''
    else:
        prev_tag = items[1].rstrip()

#归一化得到概率
P = P / sum(P)
for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])

def log(v):
    if v == 0:
        return np.log(v + 0.00001)
    return np.log(v)

#维特比算法解码，获得最优切分路径
def viterbi(x,P,A,B):
    x = [word2id[word] for word in x.split(' ')]
    T = len(x)
    dp = np.zeros((T,N)) #动态规划组
    ptr = np.zeros((T,N),dtype=int) # 保存最大路径对应的下标
    for j in range(N):
        dp[0][j] = log(P[j]) + log(B[j][x[0]])

    for i in range(1,T):#取第i个词
        for j in range(N):#计算当前词属于某个隐状态的概率
            dp[i][j] = -999999
            for k in range(N):
                score = dp[i-1][k] + log(A[k][j]) + log(B[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j] = score
                    ptr[i][j] = k

    best_sequence = [0] * T
    best_sequence[T-1] = np.argmax(dp[T-1])
    for i in range(T-2,-1,-1):
        best_sequence[i] = ptr[i+1][best_sequence[i+1]]
    print(best_sequence)
    for i in range(len(best_sequence)):
        print(id2tag[best_sequence[i]])

if __name__ == "__main__":
    x = "I like to play ball"
    viterbi(x,P,A,B)










