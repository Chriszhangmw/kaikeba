import json

train_data = open('./train_data/train.csv','w',encoding='utf-8')


with open('./train_data.json','r',encoding='utf-8') as f_r:
    data = f_r.readlines()
    f_r.close()

p_label = {}
with open('./relation_label.csv','r',encoding='utf-8') as f0:
    labels = f0.readlines()
    f0.close()
for line in labels:
    line = line.strip()
    line = line.split(',')
    p_label[line[1]] = int(line[0])

def get_index(text,entity):
    text = str(text)
    entity = str(entity)
    entity_len = len(entity)
    start = text.index(entity)
    end = start + entity_len
    return start,end


for line in data:
    line = json.loads(line.strip('\n'))
    text = line['text']
    spoList = line['spo_list']
    p_list = {}
    for spo in spoList:
        p = spo['predicate']
        p = p_label[p]
        s = spo['subject']
        o = spo['object']['@value']
        s_s,s_e = get_index(text,s)
        o_s,o_e = get_index(text,o)
        if p not in p_list.keys():
            start = []
            end = []
            start.append(s_s)
            start.append(o_s)
            end.append(s_e)
            end.append(o_e)
            p_list[p] = start,end
        else:
            start,end = p_list[p]
            start.append(s_s)
            start.append(o_s)
            end.append(s_e)
            end.append(o_e)

    final_label = []
    for k,v in p_list.items():
        s,e = p_list[k]
        temp = []
        temp.append(s)
        temp.append(k)
        temp.append(e)
        final_label.append(temp)
    sample = text + 'wenbenfengefu'  + str(final_label) + '\n'
    train_data.write(sample)





