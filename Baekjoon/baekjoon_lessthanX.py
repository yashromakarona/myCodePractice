num = input()
A = input()

num_list = num.split()
N = num_list[0] = int(num_list[0])
X = num_list[1] = int(num_list[1])

A_list = A.split()

for i in range(len(A_list)):
	A_list[i] = int(A_list[i])

for i in range(len(A_list)):
	if A_list[i] <  X:
		print(A_list[i],"" , end="")
