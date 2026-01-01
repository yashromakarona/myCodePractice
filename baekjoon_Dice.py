num = input()

dice_list = num.split()

dice_list[0] = int(dice_list[0])
dice_list[1] = int(dice_list[1])
dice_list[2] = int(dice_list[2])

if dice_list[0] == dice_list[1] == dice_list[2]:
    print(10000 + (dice_list[0] * 1000))
    
elif dice_list[0] != dice_list[1] != dice_list[2]:
    print(100 * max(dice_list[0], dice_list[1], dice_list[2]))
else:   
    for i in range(1,3):
        if dice_list(i-1) == dice_list(i) != dice_list(i+1):
            break
        print(1000 + dice_list[i] * 100)
         