exp = []
obj = []

for i in range(1000): 
	obj.append(0)
	obj[i] = input()
	exp.append(0)
	exp[i] = obj[i].split()
	print(int(exp[i][0]) + int(exp[i][1]))
