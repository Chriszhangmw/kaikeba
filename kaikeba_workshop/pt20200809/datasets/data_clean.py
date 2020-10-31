import json



train_data = open('./train_data/train.csv','w',encoding='utf-8')




with open('./train_data.json','r',encoding='utf-8') as f_r:
    data = f_r.readlines()
f_r.close()


# entity_label = {}
# with open('./entity_label.csv','r',encoding='utf-8') as f0:
#     labels = f0.readlines()
# for line in labels:
#     line = line.strip()
#     line = line.split(',')
#     # print(line)
#     entity_label[line[1]] = int(line[0])


p_label = {}
with open('./relation_label.csv','r',encoding='utf-8') as f0:
    labels = f0.readlines()
for line in labels:
    line = line.strip()
    line = line.split(',')
    # print(line)
    p_label[line[1]] = int(line[0])
# print(p_label)


# text_ner = open('./text_to_ner_name1.csv','w',encoding='utf-8')

def get_index(text,entity):
    text = str(text)
    entity = str(entity)
    entity_len = len(entity)
    start = text.index(entity)
    end = start + entity_len
    return start,end


for line in data:
    line = json.loads(line.strip('\n'))
    # print(line)
    text = line["text"]
    spoList = line["spo_list"]
    p_list = {}
    for spo in spoList:
        p = spo["predicate"]
        p = p_label[p]
        s = spo["subject"]
        s_type = spo["subject_type"]
        # label_s = entity_label[s_type]
        # text_ner.write(s + 'fengefu' + str(label_s) + '\n')
        o = spo["object"]["@value"]
        o_type = spo["object_type"]["@value"]
        # label_o = entity_label[o_type]
        # text_ner.write(o + 'fengefu' + str(label_o) + '\n')
        # print(s,o)
        s_s, s_e = get_index(text, s)
        o_s, o_e = get_index(text, o)
        if p not in p_list.keys():
            start = []
            end = []
            start.append(s_s)
            start.append(o_s)
            end.append(s_e)
            end.append(o_e)
            p_list[p] = start,end
        else:
            start, end = p_list[p]
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

    sample = text + 'wenbenfengefu' + str(final_label) + '\n'
    train_data.write(sample)





# {"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。",
#  "spo_list": [{"Combined": false, "predicate": "鉴别诊断",
#                "subject": "产后抑郁症", "subject_type": "疾病",
#                "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
#
#
# {"text": "第五节 再生障碍性贫血 再生障碍性贫血（aplastic anemia，AA，简称再障）是一种由于多种原因引起的骨髓造血功能代偿不全，"
#          "临床上出现全血细胞减少而肝、脾、淋巴结不增大的一组综合病征。接触化学因素如苯、油漆、汽油、农药等也与再障发生有关。",
#  "spo_list": [{"Combined": true, "predicate": "发病机制",
#                "subject": "再生障碍性贫血", "subject_type": "疾病",
#                "object": {"@value": "接触化学因素"}, "object_type": {"@value": "社会学"}},
#               {"Combined": true, "predicate": "临床表现",
#                "subject": "再生障碍性贫血", "subject_type": "疾病",
#                "object": {"@value": "全血细胞减少"}, "object_type": {"@value": "症状"}},
#               {"Combined": true, "predicate": "临床表现", "subject": "再生障碍性贫血",
#                "subject_type": "疾病", "object": {"@value": "不增大"}, "object_type": {"@value": "症状"}},
#               {"Combined": true, "predicate": "病因", "subject": "再生障碍性贫血",
#                "subject_type": "疾病", "object": {"@value": "骨髓造血功能代偿"},
#                "object_type": {"@value": "社会学"}},
#               {"Combined": true, "predicate": "同义词",
#                "subject": "再生障碍性贫血", "subject_type": "疾病",
#                "object": {"@value": "aplastic anemia"}, "object_type": {"@value": "疾病"}},
#               {"Combined": true, "predicate": "同义词", "subject": "再生障碍性贫血",
#                "subject_type": "疾病", "object": {"@value": "AA"}, "object_type": {"@value": "疾病"}},
#               {"Combined": true, "predicate": "同义词", "subject": "再生障碍性贫血",
#                "subject_type": "疾病", "object": {"@value": "再障"}, "object_type": {"@value": "疾病"}},
#               {"Combined": true, "predicate": "病因", "subject": "再障",
#                "subject_type": "疾病", "object": {"@value": "接触化学因素"}, "object_type": {"@value": "社会学"}},
#               {"Combined": true, "predicate": "病因", "subject": "再障",
#                "subject_type": "疾病", "object": {"@value": "苯"}, "object_type": {"@value": "社会学"}},
#               {"Combined": true, "predicate": "病因", "subject": "再障",
#                "subject_type": "疾病", "object": {"@value": "油漆"}, "object_type": {"@value": "社会学"}},
#               {"Combined": true, "predicate": "病因", "subject": "再障",
#                "subject_type": "疾病", "object": {"@value": "汽油"}, "object_type": {"@value": "社会学"}},
#               {"Combined": true, "predicate": "病因", "subject": "再障",
#                "subject_type": "疾病", "object": {"@value": "农药"}, "object_type": {"@value": "社会学"}}]}