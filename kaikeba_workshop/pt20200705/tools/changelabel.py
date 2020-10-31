


with open('./history_wordLabel.csv','r',encoding='utf-8') as f:
    data = f.readlines()

new = open('./data.csv','w',encoding='utf-8')

for line in data:
    line = line.strip()
    line = line.split(',')
    label = line[1]
    t = ''
    if label == '近代史':
        t = '1'
    elif label == '古代史':
        t = '2'
    else:
        t = '0'
    text = line[0]
    text = text.split(' ')
    text = ''.join(text)
    line_in = t + '\t' + text + '\n'
    new.write(line_in)



