a = input()
b = int(input())

c = a.split()

c[0] = int(c[0])
c[1] = int(c[1])

total_min = c[1] + b

if total_min // 60 == 0:
    c[1] = total_min
    
else:
    c[0] += (total_min // 60)
    c[1] = (total_min % 60)
    
if c[0] >= 24:
    c[0] = c[0] % 24
    
print(c[0],c[1])

