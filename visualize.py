import config
import cv2
import os
paths = os.listdir(os.path.join(config.word_data[0],'gt'))
train_imgs = os.listdir(os.path.join(config.word_data[0],'imgs'))
for path in paths:
    img = cv2.imread(os.path.join(config.word_data[0],'imgs')+'/'+path.split('.')[0].replace('gt_','')+'.jpg')
    f = open(os.path.join(config.word_data[0],'gt') +'/'+ path,'r')
    lines = f.readlines()
    for line in lines:
        arr = line.split(',')
        point1 = (int(arr[0]),int(arr[1]))
        point2 = (int(arr[2]),int(arr[3]))
        point3 = (int(arr[4]),int(arr[5]))
        point4 = (int(arr[6]),int(arr[7]))
        img = cv2.line(img,point1,point2,(0,255,0),1)
        img = cv2.line(img,point2,point3,(0,255,0),1)
        img = cv2.line(img,point3,point4,(0,255,0),1)
        img = cv2.line(img,point4,point1,(0,255,0),1)
    cv2.imwrite('./vis_vin_dataset/'+path.split('.')[0].replace('gt_','')+'.jpg',img)

# 28 58 65 1071 1016 
