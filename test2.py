from PIL import Image
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
K.set_image_dim_ordering('th')
from matplotlib import pyplot


model = load_model('my_model.h5')

imgs = os.listdir('./pics')
num = len(imgs)


def get_result(numofimg, cla):
    global imgs
    with open('./result.txt', 'a') as f:
        f.write(imgs[numofimg] + '\t' + str(cla) +'\n')
    pass


for i in range(num):
    img = image.load_img('./pics/'+ imgs[i],  target_size=(32, 32))
    #pyplot.imshow(img)
    #pyplot.show()
    img = image.img_to_array(img)
    img = img.astype('float32')
    img = img/255.0
    arr = np.expand_dims(img, axis=0)
    pre = model.predict(arr)
    pre1 = np.argmax(pre)
    pre1 = pre1.astype('int')
    print(pre1)
    get_result(i,pre1)



# get_result(imgs[0])
#img = image.img_to_array(img)

#arr = np.expand_dims(img, axis=0)
