from pt20200809.relation_extraction import predict_re
from pt20200809.entity_extraction_under_relation import predict_ner


import json
with open("./datasets/test1.json","r",encoding='utf-8') as f:
    testdata = f.readlines()


query_mapping  = json.load(open('./datasets/train_data/query_mapping.json','r',encoding='utf-8'))

p_label = {}
with open('./datasets/relation_label.csv','r',encoding='utf-8') as f0:
    labels = f0.readlines()
for line in labels:
    line = line.strip()
    line = line.split(',')
    # print(line)
    p_label[int(line[0])] = line[1]



def generate_so(ner_list):
    res = []
    for i in ner_list:
        for j in ner_list:
            if i != j:
                res.append([i,j])
    return res

n = 0
with open('./results.json','w',encoding='utf-8') as f:
    for line in testdata:
        n += 1
        line = json.loads(line.strip('\n'))
        text = line["text"]
        line["spo_list"] = []
        p_list = predict_re(text)
        for p in p_list:
            predicate_name = p_label[int(p)]
            query = query_mapping[str(p)]
            ner = predict_ner(query, text)
            # print(ner)
            res = generate_so(ner)
            for [i,j] in res:
                s = i
                o = j
                temp = {}
                temp["predicate"] =predicate_name
                temp["subject"] = s
                temp["object"] = {"@value":o}
                line["spo_list"].append(temp)
        f.write(str(line) + '\n')
        if n % 100 == 0:
            print(n)



