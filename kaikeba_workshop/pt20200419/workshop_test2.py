


sents = []
a = ['w','e','r','t']
start_p = a.index('w', 0)
end_p = a.index('t', start_p + 1)
cur = end_p + len('t')
sents.append(a[start_p + len('w'): end_p])
print(start_p)
print(end_p)
print(sents)


