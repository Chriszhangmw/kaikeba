from jieba import posseg
import jieba
import re
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import pickle
import os
# from pt20200503.utils.config import en_max_length, de_max_length
import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool
from pt20200503.utils.config import stop_word_path,train_data_path,test_data_path,\
    train_x_pad_path,train_y_pad_path,test_x_pad_path,train_x_path,train_y_path,\
    test_x_path,encoder_length,decoder_length,vocab_size,vocab_path,save_wv_model_path,\
    train_set_seg_x,train_set_seg_y,test_seg_path,embedding_matrix_path,sentence_path

from gensim.models.word2vec import LineSentence, Word2Vec

from collections import defaultdict


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        for line in f1:
            words = line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]

    return vocab, reverse_vocab





REMOVE_WORDS = ['|', '[', ']', '语音', '图片']


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines
def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)

def remove_words(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list
def segment(sentence, cut_type='word', pos=False):
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)

def parse_data(train_path, test_path):
    #读取csv
    train_df = pd.read_csv(train_path, encoding='utf-8')
    #去除report为空的
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    #剩余字段是输入，包含Brand,Model,Question,Dialogue，如果有空，填充即可
    train_df.fillna('', inplace=True)
    #实际的输入X仅仅选择两个字段，将其拼接起来
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = []
    if 'Report' in train_df.columns:
        train_y = train_df.Report
        assert len(train_x) == len(train_y)

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_y = []
    return train_x, train_y, test_x, test_y


def save_data(data_1, data_2, data_3, data_path_1, data_path_2, data_path_3, stop_words_path):
    stopwords = read_stopwords(stop_words_path)
    with open(data_path_1, 'w', encoding='utf-8') as f1:
        count_1 = 0
        for line in data_1:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                #考虑stopwords
                seg_list = [word for word in seg_list if word not in stopwords]
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f1.write('%s' % seg_line)
                    f1.write('\n')
                    count_1 += 1
        print('train_x_length is ', count_1)

    with open(data_path_2, 'w', encoding='utf-8') as f2:
        count_2 = 0
        for line in data_2:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                # seg_list = remove_words(seg_list)
                seg_list = [word for word in seg_list if word not in stopwords]
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f2.write('%s' % seg_line)
                    f2.write('\n')
                else:
                    f2.write("随时 联系")
                    f2.write('\n')
                    # print('11111')
                count_2 += 1
        print('train_y_length is ', count_2)

    with open(data_path_3, 'w', encoding='utf-8') as f3:
        count_3 = 0
        for line in data_3:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                seg_list = remove_words(seg_list)
                seg_list = [word for word in seg_list if word not in stopwords]
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f3.write('%s' % seg_line)
                    f3.write('\n')
                    count_3 += 1
        print('test_y_length is ', count_3)


def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_line = ' '.join(seg_list)
    return seg_line



def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def build(train_x_seg_path, test_y_seg_path, test_seg_path, out_path=None, sentence_path='',vocab_path = '',
          w2v_bin_path=save_wv_model_path, min_count=100):
    vocab = Vocab(vocab_path,vocab_size)
    sentences = extract_sentence(train_x_seg_path, test_y_seg_path, test_seg_path)
    save_sentence(sentences, sentence_path)
    print('train w2v model...')
    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=5)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}

    for word,index in vocab.word2id.items():
        if word in model.vocab:
            word_dict[index] = model[word]
        else:
            word_dict[index] = np.random.uniform(-0.025,0.025,(256))
    dump_pkl(word_dict,out_path,overwrite=True)




class Vocab:
    def __init__(self, vocab_file, vocab_max_size=None):
        self.PAD_TOKEN = '<PAD>'
        self.UNKNOWN_TOKEN = '<UNK>'
        self.START_DECODING = '<START>'
        self.STOP_DECODING = '<STOP>'

        self.MASK = ['<PAD>','<UNK>','<START>','<STOP>']
        self.MASK_LEN = len(self.MASK)
        self.pad_token_index = self.MASK.index(self.PAD_TOKEN)
        self.unk_token_index = self.MASK.index(self.UNKNOWN_TOKEN)
        self.start_token_index = self.MASK.index(self.START_DECODING)
        self.stop_token_index = self.MASK.index(self.STOP_DECODING)
        self.word2id,self.id2word = self.load_vocab(vocab_file,vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self,vocab_file,vocab_max_size=None):

        vocab = {mask:index for index,mask in enumerate(self.MASK)}
        reverse_vocab = {index:mask for index,mask in enumerate(self.MASK)}

        for line in open(vocab_file,'r',encoding='utf-8').readlines():
            word,index = line.strip().split('\t')
            index = int(index)
            if vocab_max_size and index > vocab_max_size - self.MASK_LEN -1:
                break
            vocab[word] = index + self.MASK_LEN
            reverse_vocab[index + self.MASK_LEN] = word
        return vocab, reverse_vocab

    def word_to_id(self,word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self,word_id):
        if word_id not in self.id2word:
            return self.UNKNOWN_TOKEN
        return self.id2word[word_id]

    def size(self):
        return self.count




























if __name__ == '__main__':

    # step 1 清洗数据
    # train_list_src, train_list_trg, test_list_src, _ = parse_data(train_data_path,
    #                                                               test_data_path)
    # save_data(train_list_src,
    #           train_list_trg,
    #           test_list_src,
    #           train_set_seg_x,
    #           train_set_seg_y,
    #           test_seg_path,
    #           stop_words_path=stop_word_path)

    # step 2 构建需要的词表
    # lines = read_data(train_set_seg_x,
    #                   train_set_seg_y,
    #                   test_seg_path)
    # vocab, reverse_vocab = build_vocab(lines)
    # save_word_dict(vocab, vocab_path)


    # step 3  构建词向量,这里是已经对应了index转化之后的词向量
    build(train_set_seg_x,
            train_set_seg_y,
            test_seg_path,
          out_path=embedding_matrix_path,
          sentence_path=sentence_path,
          vocab_path = vocab_path)







