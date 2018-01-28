from PIL import Image
import os
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot

imgs = os.listdir('./pics')
num = len(imgs)

img = image.load_img('./pics/'+ imgs[0],  target_size=(32, 32))
#pyplot.imshow(img)
#pyplot.show()
img = image.img_to_array(img)
img = img.astype('float32')
img = img/255.0


def get_result(numofimg):
    with open('./result.txt','a') as f:
        f.write(numofimg+'\t'+'1\n')
    pass

get_result(imgs[0])
#img = image.img_to_array(img)

#arr = np.expand_dims(img, axis=0)
