from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from pt20200913.config2 import Config
import numpy as np
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class MaskedLM():
    def __init__(self,topK):
        self.topK = topK
        self.tokenizer = Tokenizer(Config.BERT_VOCAB_PATH,do_lower_case=True)
        self.model = build_transformer_model(Config.BERT_CONFIG_PATH,Config.BERT_CHECKPOINT_PATH,with_mlm =True)
        self.token_ids , self.segments_ids = self.tokenizer.encode(' ')
    def tokenizer_text(self,text):
        self.token_ids,self.segments_ids = self.tokenizer.encode(text)
    #输入： 我喜欢吃程度的火锅  [5,6]
    def find_topn_candidates(self,error_index):
        for i in error_index:
            self.token_ids[i] = self.tokenizer._token_dict['[MASK]']
        probs = self.model.predict([np.array([self.token_ids]),np.array([self.segments_ids])])[0]
        print(probs)
        print(probs[5])
        for i in range(len(error_index)):
            error_id = error_index[i]
            top_k_pros = np.argsort(-probs[error_id])[:self.topK]
            candidates,fin_prob = self.tokenizer.decode(top_k_pros),probs[error_id][top_k_pros]
            print(dict(zip(candidates,fin_prob)))


if __name__ == "__main__":
    maskLm = MaskedLM(5)
    text = "刚刚一直在和老黄谈天，他和他聊天很愉快"
    maskLm.tokenizer_text(text)
    maskLm.find_topn_candidates([9,12])






