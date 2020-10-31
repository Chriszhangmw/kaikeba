import os
import numpy as np
from bert4keras.backend import keras,K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding,DataGenerator,ViterbiDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import json
import tensorflow as tf
from keras.layers import Input,Lambda
from keras.utils import to_categorical
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


ELECTRA_CONFIG_PATH = "./pre_train/electra/chinese_electra_small_L-12_H-256_A-4/electra_config.json"
ELECTRA_CHECKPOINT_PATH = "./pre_train/electra/chinese_electra_small_L-12_H-256_A-4/electra_small"
ELECTRA_VOCAB_PATH = "./pre_train/electra/chinese_electra_small_L-12_H-256_A-4/vocab.txt"


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') #space类用未经训练的【unused1】表示
            else:
                R.append('[UNK]')
        return R

maxLen = 350
epochs = 40
batch_size = 32
bert_layer = 12
learning_rate = 1e-5


model_path = "./models/ner/best_model.weights"

def load_data(filename):
    D = []
    with open(filename,encoding='utf-8') as f:
        lines = f.readlines()
    for l in lines:
        if not l:
            continue
        l = l.strip()
        l = l.split('jjjj')
        query = l[0]
        text = l[1]
        start_ids = l[2]
        end_ids = l[3]
        D.append((query,text,start_ids,end_ids))
    return D


tokenizer = OurTokenizer(ELECTRA_VOCAB_PATH,do_lower_case=True)

class data_generator(DataGenerator):

    def __iter__(self,random=False):
        batch_token_ids,batch_segment_ids,batch_start,batch_end = [],[],[],[]
        for is_end,(query,text,start_ids,end_ids) in self.sample(random):
            query_token_ids,query_segment_ids = tokenizer.encode(query)

            token_ids = query_token_ids.copy()
            start = query_segment_ids.copy()
            end = query_segment_ids.copy()

            w_token_ids = tokenizer.encode(text)[0][1:-1]
            text_len = len(w_token_ids)
            if len(token_ids) + text_len < maxLen:
                token_ids += w_token_ids
                start_tmp = [0] * text_len
                end_tmp = [0] * text_len
                try:
                    for s_idx in start_ids:
                        start_tmp[int(s_idx)] = 1
                    for e_idx in end_ids:
                        end_tmp[int(e_idx)] = 1
                except:
                    print('index error')
                start += start_tmp
                end += end_tmp
            else:
                continue
            token_ids += [tokenizer._token_end_id]
            query_segment_ids = query_segment_ids + [1] * (len(token_ids) - len(query_segment_ids))
            start += [0]
            end += [0]

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(query_segment_ids)
            batch_start.append(to_categorical(start,2))
            batch_end.append(to_categorical(end,2))

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_start = sequence_padding(batch_start)
                batch_end = sequence_padding(batch_end)
                yield [batch_token_ids,batch_segment_ids,batch_start,batch_end],None
                batch_token_ids, batch_segment_ids, batch_start, batch_end = [],[],[],[]

bert_model = build_transformer_model(
config_path=ELECTRA_CONFIG_PATH,
    checkpoint_path=ELECTRA_CHECKPOINT_PATH,
    model='eletra'
)

mask = bert_model.input[1]
start_labels = Input(shape=(None,2),name="start_labels")
end_labels = Input(shape=(None,2),name="end_labels")


out_layers = ""



def extract(qtext):
    v = qtext.split('jjjjj')[0]
    text = qtext.split('jjjjj')[1]

    query_tokens,query_segment_ids = tokenizer.encode(v)
    token_ids = query_tokens.copy()
    token_ids_w = tokenizer.encode(text)[0][1:-1]
    token_ids += token_ids_w
    token_ids += [tokenizer._token_end_id]
    query_ids = query_segment_ids + [1]*(len(token_ids) - len(query_segment_ids))

    start_out = start_model.predict([[token_ids],[query_ids]])[0][len(query_segment_ids):-1]
    end_out = end_model.predict([[token_ids],[query_ids]])[0][len(query_segment_ids):-1]

    start = np.argmax(start_out,axis=1)
    end = np.argmax(end_out, axis=1)
    res = [int(k) + int(v) if int(k) + int(v) < 2 else 1 for k,v in zip(start,end)]

    return res



class Evaluate(keras.callbacks.Callback):
    def __init__(self,val):
        super().__init__()
        self.best_val_f1 = 0
        self.val = val
    def eveluate(self):
        X,Y,Z = 1e-10,1e-10,1e-10




def predict_ner(query,text):


