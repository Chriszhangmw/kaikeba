datapath = './baidu_95.csv'

with open(datapath,'r',encoding='utf-8') as f:
    data = f.readlines()
f.close()

labels = open('./label.csv','w',encoding='utf-8')
temp = set()
for line in data:
    line = line.strip()
    line = line.split(",")
    kgs = line[0]
    kgs = kgs.split(' ')
    for kg in kgs:
        temp.add(kg)

temp = list(temp)
for index,kg in enumerate(temp):
    labels.write(str(index) + ',' + kg + '\n')
