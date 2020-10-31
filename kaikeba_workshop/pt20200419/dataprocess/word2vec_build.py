from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import pickle
import os

def dump_pkl(vocab, pkl_path, overwrite=True):
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("save %s ok." % pkl_path)

def read_lines(path):
    lines = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    print(lines)
    lines += read_lines(train_y_seg_path)
    print(lines)
    lines += read_lines(test_seg_path)
    print(lines)
    for line in lines:
        ret.append(line)
        print(line)
    print(ret)
    return  ret



def save_sentence(lines, sentence_path):
    with open(sentence_path,'w',encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    f.close()


def build(train_x_seg_path, test_y_seg_path, test_seg_path, out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=200):
    sentences = extract_sentence(train_x_seg_path,test_y_seg_path,test_seg_path)
    save_sentence(sentences,sentence_path)

    w2v = Word2Vec(sentences=LineSentence(sentence_path),size=256,window=5,min_count=200,iter=5)
    w2v.wv.save_word2vec_format(w2v_bin_path,binary=True)

    model = KeyedVectors.load_word2vec_format(w2v_bin_path,binary=True)

    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict,out_path,overwrite=True)







if __name__ == '__main__':
    build('../data/train_set.seg_x.txt',
          '../data/train_set.seg_y.txt',
          '../data/test_set.seg_x.txt',
          out_path='../data/word2vec.txt',
          sentence_path='../data/sentences.txt')