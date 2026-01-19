code = input()
code_list = code.split()

for i in range(5):
    code_list[i] = int(code_list[i])

code_list.append(0)
for i in range(5):
    code_list[5] += (code_list[i]**2)

code_list[5] = code_list[5] % 10
print(code_list[5])