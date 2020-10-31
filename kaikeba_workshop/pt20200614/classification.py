
import  numpy  as np
import jieba
import random
import re

#根据预料来构建词表（非重复）
def createVoclabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return  list(vocabSet)

#根据词表，来向量化输入的样本，假设词表的长度是N， 向量化的结果是[1,n]
def setOfwords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for index,word in enumerate(vocabList):
        if word in inputSet:
            returnVec[index] = 1
    # for word in inputSet:
    #     if word in vocabList:
    #         returnVec[vocabList.index(word)] = 1
    return returnVec



def train(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)#样本的个数
    numWords = len(trainMatrix[0]) #每个文档的词数
    trainMatrix = np.array(trainMatrix)
    trainCategory = np.array(trainCategory)

    pB = sum(trainCategory)/numTrainDocs
    pA = 1 - pB

    pa = np.ones(numWords)
    pb = np.ones(numWords)
    pa_ = 2.0
    pb_ = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            pb += trainMatrix[i]
            pb_ += sum(trainMatrix[i])
        else:
            pa += trainMatrix[i]
            pa_ += sum(trainMatrix[i])

    pa = np.log(pa/pa_)
    pb = np.log(pb/pb_)

    return pa, pb, pA








def classifyBN(inputMatrix, pa, pb, pB):
    p1 = sum(inputMatrix * pa) + np.log(1-pB)
    p2 = sum(inputMatrix * pb) + np.log(pB)
    if p1 > p2:
        return  0
    else:
        return 1





def textParse(text):
    line = re.sub(r'[a-zA-z.【】0-9、。，/！...~*\n]','',text)
    listOfTockens = jieba.cut(line,cut_all=False)
    return [tok.lower() for tok in listOfTockens if len(tok) > 1]

def classification():
    docList = []
    classList = []
    for i in range(1,11):
        wordList_neg = textParse(open('./data/spam/%d.txt' % i ,'r',encoding='gb2312',errors='ignore').read())
        docList.append(wordList_neg)
        classList.append(1)
        wordList_pos = textParse(open('./data/ham/%d.txt' % i ,'r',encoding='gb2312',errors='ignore').read())
        docList.append(wordList_pos)
        classList.append(0)
    vocabList = createVoclabList(docList)

    trainSet = list(range(20))
    testSet = []
    for _ in range(5):
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfwords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])

    pa,pb,pA = train(trainMat,trainClass)


    count_eorr = 0
    for docIndex in testSet:
        wordVec = setOfwords2Vec(vocabList,docList[docIndex])
        result  = classifyBN(wordVec,pa,pb,pA)
        print('*'*20)
        print('prediction results:',result)
        print('true results ',classList[docIndex])
        print('*' * 20)
        if result != classList[docIndex]:
            count_eorr +=1

    print(count_eorr)

if __name__ == '__main__':
    classification()















