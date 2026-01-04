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
    if dice_list[0] == dice_list[1] != dice_list[2]:
        print(1000 + dice_list[1] * 100)
        
    elif dice_list[0] != dice_list[1] == dice_list[2]:
        print(1000 + dice_list[1] * 100)
        
    elif dice_list[0] == dice_list[2] != dice_list[1]:
        print(1000 + dice_list[2] * 100)