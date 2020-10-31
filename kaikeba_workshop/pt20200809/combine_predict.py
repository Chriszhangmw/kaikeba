

import json
from pt20200809.relation_extraction import predict_re
from pt20200809.entity_extraction_under_relation import predict_ner

query_mapping = json.load(open('./query_mapping.json','r',encoding='utf-8'))

def run(text):
    relation = predict_re(text)
    res = {}
    for r in relation:
        s = str(r)
        query = query_mapping[r]
        ner = predict_ner(query,text)
        res[query] = ner
    return res

if __name__ == "__main__":
    text = '发病以来，精神差，饮食可，大小便正常'
    res = run(text)
    for k,v in res.items():
        print(u'样本存在关系： %s, 在该关系下存在的实体有： %s' % (k,v))












