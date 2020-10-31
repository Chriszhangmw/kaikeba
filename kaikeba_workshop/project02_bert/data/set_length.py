import matplotlib.pyplot as plt

datapath = './baidu_95.csv'

with open(datapath,'r',encoding='utf-8') as f:
    data = f.readlines()
f.close()

lensentencetitle = {}
lensentencecontent = {}
for line in data:
    line = line.strip()
    line = line.split(",")
    title = line[0]

    content = line[1]
    lens2 = len(title)
    lens1 = len(content)


    if lens1 not in lensentencecontent.keys():
        lensentencecontent[lens1] = 1
    else:
        lensentencecontent[lens1] += 1

plt.bar(lensentencecontent.keys(),lensentencecontent.values())
plt.show()

