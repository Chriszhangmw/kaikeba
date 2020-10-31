
import json

f_w = open('train_data/relation_datasets.csv','w',encoding='utf-8')

f_w_ner = open('train_data/ner_datasets.csv','w',encoding='utf-8')
query_mapping = json.load(open('./train_data/query_mapping.json','r',encoding='utf-8'))

with open('./train_data/train.csv','r',encoding='utf-8') as f_r:
    data = f_r.readlines()
    f_r.close()

for line in data:
    line = line.strip()
    line = line.split('wenbenfengefu')
    text = line[0]
    spoList = line[1]
    spoList = json.loads(spoList)
    if spoList == []:
        continue
    label_list = []
    re_ner = {}
    for spo in spoList:
        p = str(spo[1])
        s_ids = spo[0]
        e_ids = spo[2]
        for i in range(len(s_ids)//2):
            s_start = s_ids[2*i]
            o_start = s_ids[2*i + 1]
            s_end = e_ids[2*i]
            o_end = e_ids[2*i + 1]
            if p not in label_list:
                label_list.append(p)
                start_ner_idx = []
                end_ner_idx = []
                start_ner_idx.append(str(s_start))
                start_ner_idx.append(str(o_start))
                end_ner_idx.append(str(s_end))
                end_ner_idx.append(str(o_end))
                re_ner[p] = [start_ner_idx,end_ner_idx]
            else:
                start_ner_idx = re_ner[p][0]
                end_ner_idx = re_ner[p][1]
                start_ner_idx.append(str(s_start))
                start_ner_idx.append(str(o_start))
                end_ner_idx.append(str(s_end))
                end_ner_idx.append(str(o_end))
    label_string = ' '.join(label_list)
    relation_line = text + 'jjjj' + label_string + '\n'
    f_w.write(relation_line)
    for k,v in re_ner.items():
        p_query = query_mapping[k]
        start_ner_idx = v[0]
        end_ner_idx = v[1]
        assert len(start_ner_idx) == len(end_ner_idx)
        ner_line = p_query + 'jjjj' + text + 'jjjj' + ' '.join(start_ner_idx) + 'jjjj' + ' '.join(end_ner_idx) + '\n'
        f_w_ner.write(ner_line)




















