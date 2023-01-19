import config
import os
import shutil
print(config.word_data)

path = os.listdir(os.path.join(config.word_data[0],'labels'))
train= os.listdir(os.path.join(config.word_data[0],'imgs'))
print(path[2])
arr = []
for item in train:
    txt = item.split('.')[0]
    num = int(txt.replace('im',''))
    arr.append(num)
a = os.path.join(config.word_data[0],'labels') + '/gt_' + str(arr[2]) + '.txt'
print(os.path.exists(a))
for i in range(len(arr)):
    src = os.path.join(config.word_data[0],'labels') + '/gt_' + str(arr[i]) + '.txt'
    des = os.path.join(config.word_data[0],'gt') + '/gt_im' + str(arr[i]).zfill(4) + '.txt'
    if os.path.exists(src):
        shutil.copyfile(src,des)
print(len(os.listdir(os.path.join(config.word_data[0],'gt'))))
#shutil.copyfile(os.path.join(config.word_data[0]+'gt',path[0]),os.path.join(config.word_data[0]+'gt','gt_im'+path[0].split('_')[1]))

