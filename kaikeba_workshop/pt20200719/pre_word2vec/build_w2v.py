import pathlib
import os
import time
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence
import numpy as np
import pickle
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
def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def train_w2v_model(txtPath,model_path):
    print(txtPath)
    w2v_model = Word2Vec(LineSentence(txtPath), workers=4,min_count=1)  # Using 4 threads
    w2v_model.save(model_path) # Can be used for continue trainning
    # w2v_model.wv.save('w2v_gensim') # Smaller and faster but can't be trained later


def build_embedding(vocab_path):
    vocab = open(vocab_path, encoding='utf-8').readlines()
    vocab_size = len(vocab)
    print(vocab_size)
    embedding_dim = 300
    vocab_dict = {}
    for i,w in enumerate(vocab):
        w = w.strip()
        w = w.split()
        vocab_dict[w] = str(i)

    # vocab_dict = {w.strip().split():i for i,w in enumerate(vocab)}
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    out_path = './word2vec.txt'
    word2vec_dict =load_pkl(out_path)
    embedding_dim = 300

    for word in vocab:
        word = word.strip()
        word = word.split()
        i = vocab_dict[word]
        embedding_vector = word2vec_dict[word]
        try:
            embedding_matrix[int(i)] = embedding_vector
        except:
            embedding_matrix[int(i)] = np.random.uniform(-0.0025, 0.0025, (embedding_dim))
    np.savetxt('./embedding.txt', embedding_matrix, fmt='%0.8f')


if __name__ == '__main__':

    # w_in = open('./vocab.txt','w',encoding='utf-8')
    # with open('./baidu_95.txt','r',encoding='utf-8') as f:
    #     data = f.readlines()
    # f.close()
    #
    # for line in data:
    #     line = line.strip()
    #     line = line.split(' ')
    #     w_in.write(line)
    # w_in.close()
    pwd_path = pathlib.Path(os.path.abspath(__file__)).parent.parent
    root_path = os.path.join(pwd_path, 'pre_word2vec')
    print(root_path)
    txt_path = os.path.join(root_path, "baidu_95.txt")
    # txtPath = './vocab.txt'
    model_path = './w2v.model'
    out_path = './word2vec.txt'
    # train_w2v_model(txt_path, model_path)

    model = Word2Vec.load(model_path)
    print(len(model.wv.vocab))
    # print(model['正确'])
    out_path2 = './word2vec.txt'
    word2vec_dict = load_pkl(out_path2)
    print(len(word2vec_dict))
    # print(word2vec_dict['基础产业'])


    # word_dict = {}
    # for word in model.wv.vocab:
    #     word_dict[word] = model[word]
    # dump_pkl(word_dict, out_path, overwrite=True)

    vocab_path = './baidu_95_vocab.txt'
    build_embedding(vocab_path)







