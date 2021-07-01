import os
import re
from numpy.core.arrayprint import DatetimeFormat
# import cv2
import numpy as np
from PIL import Image
data = np.empty(shape=(0,1))
target = []
def eachFile(filepath):
    flag = True
    i = -1
    for root, dirs, files in os.walk(filepath):        
        for file in files:
            if re.search('Ambient', file):
                continue
            if re.search('.pgm',file):
                #print(os.path.join(root, file))
                im = Image.open(os.path.join(root, file))
                
                # 获得图像尺寸:
                w, h = im.size
                # 缩放到25%:
                im.thumbnail((w//4, h//4))

               
                img = np.array(im).flatten()
                if  flag:
                    flag = False
                    data = img
                    target.append(i)
                else:
                    data =  np.vstack((data,img))
                    target.append(i)
                
    
        i=i+1
    tmp = np.array(target)[np.newaxis,:]
    print(tmp)

    data = np.concatenate((data,tmp.T),axis =1)
    np.savetxt("new.csv", data, delimiter=',')          
if __name__ == '__main__':
    filepath = r'C:\Users\LiQi\Desktop\yaleBExtData'
    eachFile(filepath)