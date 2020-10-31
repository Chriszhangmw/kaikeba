import sys, pickle, os, random
import numpy as np

sopwords = [',','.','<','>',';','"','"','!','%','*','&','?','/','。','，','(',')','（','）','【','】','[',']','{','}','+','=','-']

def read_corpus_test(corpus_path):
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_ = []
    for line in lines:
        if line != '\n':
            # print(line)
            tmp = line.strip().split(' ')
            # print(tmp)
            char = tmp[0]
            sent_.append(char)
        else:
            data.append((sent_))
            sent_ = []
    return data


def sentence2id(sent, word2id):
    sentence_id = []
    delete_word = []
    for word in sent:
        if word not in word2id:
            # delete_word.append(word)
            sentence_id.append(0)
        else:
            sentence_id.append(word2id[word])
            # continue

    return sentence_id,delete_word


def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id

def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def batch_yield_test(data, vocab):
    seqs,sample = [],[]

    for sent_ in data:
        sent_id,delete_word = sentence2id(sent_, vocab)
        seqs.append(sent_id)
        sample.append(sent_)
    return seqs, sample
    # if len(seqs) != 0:
    #     yield seqs,sample




























