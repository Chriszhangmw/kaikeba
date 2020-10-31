from collections import defaultdict

def save_word_dict(vocab, save_path):
    pass

def read_data(path_1, path_2, path_3):
    pass


def build_vocab(items, sort=True, min_count=0, lower=False):
    pass


if __name__ == '__main__':
    lines = read_data('../data/train_set.seg_x.txt',
                      '../data/train_set.seg_y.txt',
                      '../data/test_set.seg_x.txt')
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, '../data/vocab.txt')


