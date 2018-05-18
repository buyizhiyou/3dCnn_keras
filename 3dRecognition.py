#-*-coding:utf8-*-

# Source: http://learnandshare645.blogspot.hk/2016/06/3d-cnn-in-keras-action-recognition.html

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.callbacks import TensorBoard

import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pdb

from preprocess_data import read_data

X_tr = read_data()
X_train = np.array(X_tr)   # convert the frames read into array
num_samples = len(X_train)#600
# Assign Label to each class (label is a 1-D array of 0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3, etc)
label=np.ones((num_samples,), dtype = int)
#pdb.set_trace()
label[0:100] = 0
label[100:200] = 1
label[200:300] = 2
label[300:400] = 3
label[400:500]= 4
label[500:] = 5
y_train = label

#600 num_samples
img_rows,img_cols,img_depth=32,32,15
print('X_Train shape:', X_train.shape)#(600, 32, 32, 15)
train_set = np.zeros((num_samples, img_rows, img_cols, img_depth, 1))

for h in range(num_samples):
    train_set[h,:,:,:,0]=X_train[h,:,:,:]

patch_size = 15    # img_depth or number of frames used for each video
print(train_set.shape, 'train samples')
# CNN Training parameters
batch_size = 4
nb_classes = 6
nb_epoch = 50
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
# number of convolutional filters to use at each layer
nb_filters = [  32,   # 1st conv layer
                64    # 2nd
             ]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5,5]

#Define optimizer
RMSprob = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#Load or Define model
model_path = './models/2018-05-18 10:41:36-model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("**************************************************")
    print("model loaded")

else:
    model = Sequential()
    print('input shape', img_rows, 'rows', img_cols, 'cols', patch_size, 'patchsize')
    model.add(Conv3D(nb_filters[0],(5,5,5),input_shape=(img_rows, img_cols,patch_size,1),activation='relu'))
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
    model.add(BatchNormalization(momentum=0.99))
    model.add(Conv3D(nb_filters[1],(3,3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])
model.summary()
# Split the data
X_train_new, X_val_new, y_train_new,y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)
# Train the model
hist = model.fit(X_train_new,
			    y_train_new,
			    validation_data=(X_val_new,y_val_new),
			    batch_size=batch_size,
			    epochs = nb_epoch,
			    shuffle=True,
			    callbacks=[TensorBoard(log_dir='./log')] )#tensorboard --logdir=./log

# Save model
now = str(datetime.datetime.now()).split('.')[0]
model.save('./models/'+now+"-model.h5")
# Evaluate the model
score = model.evaluate(X_val_new,y_val_new,batch_size=batch_size)
# Print the results
print('**********************************************')
print('Test score:', score)
print('History', hist.history)





