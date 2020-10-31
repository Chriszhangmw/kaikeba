
import re,collections


def words(text):
    return re.findall("[a-z]+",text.lower())

def train(words):
    model = collections.defaultdict(int)
    for word in words:
        model[word] += 1
    return model


word_list = train(words(open('./data.txt').read()))

def know(words):
    return set(w for w in words if w in word_list)


alphabet = "abcdefghijklmnopqrstuvwxyz"
# print(word_list)
def edist1(word):
    n = len(word)
    #删除
    s1 = [word[0:i] + word[i+1:] for i in range(n)]
    #相邻的字母替换
    s2 = [word[0:i] + word[i+1]+word[i] + word[i+2:] for i in range(n-1)]
    #从26个字母表里选择一个做替换
    s3 = [word[0:i] + c + word[i+1:] for i in range(n) for c in alphabet]
    #插入，从26个英文字母里面做插入
    s4 = [word[0:i] + c + word[i:] for i in range(n) for c in alphabet]
    edits_words = set(s1 + s2 + s3 + s4)
    edits_words = know(edits_words)
    return edits_words

def edits2(word):
    edits2_words = set(e2 for e1 in edist1(word) for e2 in edist1(e1))
    edits2_words = know(edits2_words)
    return edits2_words


def correct(word):
    if word not in word_list:
        candidates = edist1(word) or edits2(word)
        print(candidates)
        return max(candidates, key=lambda w:word_list[w])
    else:
        return word


print(correct("firr"))




