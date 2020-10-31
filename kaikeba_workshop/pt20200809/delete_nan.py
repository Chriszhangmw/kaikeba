
import json

with open("./results.json","r",encoding='utf-8') as f1:
    testdata = f1.readlines()
with open('./results_new.json','w',encoding='utf-8') as f:
    for line in testdata:
        line = json.loads(line)
        # print(line)
        text = line["text"]
        spo_list = line["spo_list"]
        if spo_list == []:
            continue
        f.write(str(line) + '\n')




