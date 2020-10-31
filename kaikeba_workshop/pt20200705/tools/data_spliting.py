

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('./data.csv','r',encoding='utf-8')

rate = [0.8,0.1,0.1]
train = data[:int(rate[0] * len(data))]
dev = data[int(sum(rate[:2]) * len(data)):]
test = data[int(rate[0] * len(data)):int(sum(rate[:2] * len(data)))]

# train = open('./data/train.csv','w',encoding='utf-8')
# valid = open('./data/dev.csv','w',encoding='utf-8')
# test = open('./data/test.csv','w',encoding='utf-8')

train.to_csv('./data/train.csv',index=False,encoding='utf-8')
test.to_csv('./data/test.csv',index=False,encoding='utf-8')
dev.to_csv('./data/dev.csv',index=False,encoding='utf-8')









