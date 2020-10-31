import re
import pandas as pd
import csv
from tensorflow.keras import preprocessing
import numpy as np
import json
import jieba
from sklearn.preprocessing import MultiLabelBinarizer
import os
from pathlib import Path





def get_data(file_x, file_y, origin_data='./data/baidu_95.csv'):
    if not os.path.exists(file_x):
        preprocess(origin_data)

    x = np.load(file_x)
    y = np.load(file_y)

    return x, y

def preprocess(data_file, save_dir='../datasets/', vocab_size=200, padding_size=128):
    """
    Text to sequence, compute vocabulary size, padding sequence.
    Return sequence and label.
    """
    data_file = Path(data_file)
    print("Loading data from {} ...".format(data_file))
    df = pd.read_csv(data_file, header=None, names=["labels", "item"], dtype=str)

    # Texts to sequences
    df['item'] = df.item.apply(lambda x: list(jieba.cut(x)))
    corpus = df.item.tolist()
    text_preprocesser = preprocessing.text.Tokenizer(num_words=vocab_size)#, oov_token="<UNK>")
    text_preprocesser.fit_on_texts(corpus)
    x = text_preprocesser.texts_to_sequences(corpus)
    word_dict = text_preprocesser.word_index
    # save word2id
    with open(save_dir+'voab.txt', 'w', encoding='UTF8') as f:
        for k,v in word_dict.items():
            f.write(f'{k}\t{str(v)}\n')
    # json.dump(word_dict, open(vocab_file, 'w'), ensure_ascii=False)
    # max_doc_length = max([len(each_text) for each_text in x])
    x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                             padding='post', truncating='post')
    print("Find words: {:d}".format(len(word_dict)))
    print("Vocabulary size: {:d}".format(vocab_size))
    print("Shape of train data: {}".format(np.shape(x)))

    y = df.labels.apply(lambda x: set(x.split())).tolist()
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)

    # save label
    with open(f'{save_dir}labels_{data_file.stem}.txt', 'w', encoding='utf8') as f:
        for label in mlb.classes_:
            f.write(f'{label}\n')

    # save train test
    index = list(range(len(x)))
    np.random.seed(0)
    np.random.shuffle(index)

    x = x[index]
    y = y[index]

    split = int(len(x) * 0.9)

    np.save(f'{save_dir}{data_file.stem}_train_x.npy', x[:split])
    np.save(f'{save_dir}{data_file.stem}_test_x.npy', x[split:])
    np.save(f'{save_dir}{data_file.stem}_train_y.npy', y[:split])
    np.save(f'{save_dir}{data_file.stem}_test_y.npy', y[split:])

    print(f"Save train and test data: {save_dir}{data_file.stem}")




if __name__ == '__main__':
     preprocess('../datasets/baidu_95.csv')