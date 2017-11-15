fid = open('/home/mohsen/Desktop/MV3D/devkit_object/mapping/train_rand.txt', "r")
lines = fid.readlines()
list = []
for i in lines[0].split(','):
    list.append(i)
count = 0
for i in list:
    if i == '1':
        print(count)
    else:


        count = count +1


