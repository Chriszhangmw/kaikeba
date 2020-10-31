import tensorflow as tf
import numpy as np
import os
import os, argparse, time, random
from pt20200919.model import BiLSTM_CRF
from pt20200919.data import read_corpus, read_dictionary, tag2label, random_embedding


## Session configuration
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
# config = tf.ConfigProto()

## hyperparameters

word2id = read_dictionary('./vocab.pkl')
embeddings = random_embedding(word2id,128)
train_data = read_corpus('./c.txt')
model = BiLSTM_CRF(embeddings,tag2label,word2id,2,2,128,True,True,True)
model.build_graph()
model.train(train_data)


model.test("患者于半月前无明显诱因出现进食后中上腹不适,"
           "每次持续数分钟自行缓解,无恶心，呕吐，反酸、"
           "嗳气、烧心，无腹痛、腹胀、腹泻、便秘，无厌油、"
           "纳差，未予重视，未特殊处理。，半前至我院门诊"
           "行胃镜检查提示：浅表性胃窦炎伴糜烂，十二指肠球炎，"
           "，腹部彩超：肝回声增多，胆囊息肉样变。为进一步诊治，"
           "门诊“胃炎”收入我科。患者本次发病以来，食欲"
           "正常， 神志清醒，精神尚可，睡眠尚可，大便正常，"
           "小便正常，体重无明显变化。")




