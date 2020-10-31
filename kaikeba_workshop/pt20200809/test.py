


import json

with open("./test.json", "r", encoding='utf-8') as f:
   testdata = f.readlines()


with open("./jj.json",'w',encoding='utf-8') as f:
   new = []
   for line in testdata:
      line = json.loads(line.strip('\n'))
      line["spo_list"] = [{"p":"诊断"}]
      f.write(str(line) + '\n')





#
# def generate_so(ner_list):
#    res = []
#    for i in ner_list:
#       for j in ner_list:
#          if i != j:
#             res.append([i, j])
#    return res
#
# ner_list = ["a","b","c","d"]
#
# res = generate_so(ner_list)
#
# for [i,j] in res:
#    print(i,j)
import tqdm
import json
# with open("./datasets/test1.json","r",encoding='utf-8') as f:
#     testdata = f.readlines()


