# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# sudo apt-get update
# sudo apt-get install libgtk2.0-dev
import os
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

### Parameters
# Input file
data_directory = './data1/'
# Output directory
out_directory = './test'
# Image augmentation
st_offset_pix = 0.005  # steering angle offset per pixel translation
st_clr_offset = np.array([0,1,-1])*0.22  # steering angle offset for left, center, right
trans_max = 100  # max image translation
p = 0.9  # probability of translating the image
st_thr = 0.005  # st angle threshold
# Model
do_rate = 0.5   # dropout rate
l_reg = 0.01 # L2 regularization parameter
model ='NV'
# Training
BATCH_SIZE = 32
fact = 3  # nb_samples = factor*len(samples)
EPOCHS = 6
sel_spe = 1  # indice within 'samples_per_epoch' list to use

### Code
## Create a directory for outputs
os.makedirs(out_directory)

## Read the input csv file
samples = []
cvs_filename = data_directory+'driving_log.csv'

with open(cvs_filename) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

from sklearn.utils import shuffle

## Sample generator
def generator(samples, directory, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples_shuf = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples_shuf[offset:offset+batch_size] # selection of batch_size samples
            images = []
            angles = []
            for batch_sample in batch_samples:
                # we randomly select one of the 3 images (left, right, center)
                i_sel = np.random.randint(3)
                name = directory+'/IMG/'+batch_sample[i_sel].split('\\')[-1].split('/')[-1]
                image = cv2.imread(name)
                # convert BRG->RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # retrieve steering angle
                st_angle = float(batch_sample[3])
                # Image augmentation
                # 1-randomly translate the image
                # Translate the image : randomly for steering_angles less than a thr
                apply_trans = (np.random.rand()>=(1-p)) & (abs(st_angle)<st_thr)
                if apply_trans:
                    trans_val =np.random.uniform(-1,1)*trans_max # >0 -> left
                    M = np.array([[1,0,trans_val],[0,1,0]])
                    image = cv2.warpAffine(image,M,image.shape[1::-1])
                    # steer angle correction has same sign as trans_value
                    st_angle = st_angle+trans_val*st_offset_pix
                
                # steer angle correction for left and right images
                st_angle += st_clr_offset[i_sel]
                
                # 2-randomly flip the image
                apply_flip = np.random.rand()
                if apply_flip>0.5:
                    image = cv2.flip(image,flipCode=1)
                    st_angle = -st_angle
               
                images.append(image)
                angles.append(st_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train

## Building a model 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

if model=='initial':
    model = Sequential()
    model.add(Lambda (lambda x: (x/255.0-0.5),input_shape=(160,320,3))) # normalisation
    model.add(Cropping2D(cropping=((55, 25), (0, 0))))
    model.add(Conv2D(16,5,5,activation='relu', W_regularizer=l2(l_reg)))
    model.add(Conv2D(32,5,5,activation='relu', W_regularizer=l2(l_reg)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64,5,5,activation='relu', W_regularizer=l2(l_reg)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(do_rate))
    model.add(Dense(120,activation='relu', W_regularizer=l2(l_reg)))
    model.add(Dropout(do_rate))
    model.add(Dense(80,activation='relu', W_regularizer=l2(l_reg)))
    model.add(Dropout(do_rate))
    model.add(Dense(1))
elif model=='NV':
    model = Sequential()
    model.add(Lambda (lambda x: (x/255.0-0.5),input_shape=(160,320,3))) # normalisation
    model.add(Cropping2D(cropping=((55, 25), (0, 0)))) # 0 - (80, 320, 3)
    model.add(Dropout(do_rate))
    model.add(Conv2D(24,5,5,subsample=(2,2),activation='elu', W_regularizer=l2(l_reg))) # 1 - (38, 158, 24)
    model.add(Conv2D(36,5,5,subsample=(2,2),activation='elu', W_regularizer=l2(l_reg))) # 2 - (17, 77, 36)
    model.add(Dropout(do_rate))
    model.add(Conv2D(48,5,5,subsample=(2,2),activation='elu', W_regularizer=l2(l_reg))) # 3 - (7, 37, 48))
    model.add(Conv2D(64,5,5,activation='elu', W_regularizer=l2(l_reg))) # 4 - (3, 33, 64)
    #model.add(Dropout(do_rate))
    model.add(Conv2D(64,3,3,activation='elu', W_regularizer=l2(l_reg))) # 5 - (1, 31, 64))
    model.add(Flatten())
    model.add(Dense(100,activation='elu', W_regularizer=l2(l_reg))) # 6 - (1984)
    model.add(Dropout(do_rate))
    model.add(Dense(50,activation='elu', W_regularizer=l2(l_reg))) # 7
    #model.add(Dropout(do_rate))
    model.add(Dense(10,activation='elu', W_regularizer=l2(l_reg))) # 8
    model.add(Dense(1))

# Compile and train the model using the generator function
train_generator = generator(train_samples, data_directory, BATCH_SIZE)
validation_generator = generator(validation_samples, data_directory, BATCH_SIZE)

samples_per_epoch = [fact*len(samples),2*BATCH_SIZE] # number of samples per epoch to use

model.compile(loss='mse',optimizer='SGD')
#saves the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath=out_directory+'/weights{epoch:02d}.hdf5', verbose=1, save_best_only=False, save_weights_only=False)
history_object = model.fit_generator(train_generator, samples_per_epoch= 
            samples_per_epoch[sel_spe], validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=EPOCHS, callbacks=[checkpointer])
model.save(out_directory+'/model.h5')
### plot the training and validation loss for each epoch
plt.figure()
plt.plot(np.arange(EPOCHS)+1,history_object.history['loss'])
plt.plot(np.arange(EPOCHS)+1,history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.xlim([0.5,EPOCHS+0.5]), plt.ylim([0,0.1])
plt.savefig(out_directory+'/loss.jpg')
