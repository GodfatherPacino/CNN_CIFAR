from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print (y_train[0])
X_train = X_train.astype('float32')
X_train = X_train / 255.0

# print('next')
#print(X_train[9].shape)
# print(y_train[22])
# img1 = toimage(X_train[22])
# pyplot.imshow(img1)
# pyplot.show()
# for i in range(30):
#     if y_train[i] == 1:
#         pyplot.imshow(toimage(X_train[i]))
#         print(i)
#         pyplot.show()
#         break


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# print (X_train[0].shape)
# print (y_test.shape)

imgpath = 'img-0003.jpg'
img = image.load_img(imgpath, target_size=(32, 32))
img = image.img_to_array(img)
img = img.astype('float32')
img = img/255.0
# print(img.shape)
arr = np.expand_dims(img, axis=0)

#arr1 = arr
#print (arr.shape)

#pyplot.imshow(toimage(arr))
#pyplot.show()
#arr = X_train[0]

# print (arr.shape)

arr1 = X_train[0]
arr1 = np.expand_dims(arr1, axis=0)
model = load_model('my_model.h5')
pre = model.predict(arr)
print (pre)
pre1 = np.argmax(pre)
print (pre1)