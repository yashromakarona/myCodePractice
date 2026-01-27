k = []
l = []

t = int(input())

for i in range(t):
	k.append(0)
	l.append(0)
	k[i] = input()
	l[i] = k[i].split()

for i in range(t):
	print(int(l[i][0])+int(l[i][1]))

