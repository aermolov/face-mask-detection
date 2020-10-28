# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:37:35 2020

@author: aermolov
Here classification model is trained to detect face mask on a person.
Training data are about 3600 images with people wearing masks
and about 3700 images without masks

Function "face_extract" extracts and saves in separate file a face from original images
Fuction "resize" resizes images to the same resolution, create labels and saves images and labels as numpy vectors

20% of the images were used as a validation data
Augmentation was applied to the train dataset 
Keras sequential structure is used to train the the model. The last "Dense" layer has sigmoid activation function
The model was trained over 60 epochs, over 98% accuracy on the validation set was achieved which was enough to use in live mask recognition task

"""
#%%
import cv2
import numpy as np
from pathlib import Path
import random

#%%
#pth=Path(r'C:\Users\aermolov\Documents\ML_data\probearbeit_masks\masks_raw') # folder to take images from
#pth_proc=Path(r'C:\Users\aermolov\Documents\ML_data\probearbeit_masks\test') # folder to save processed images
def face_extract(source, destination): ### Function to crop face from picture
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Import haarcascade classifier for image face recognition
    
    y_c=10
    x_c=10
    k=0
    pth=Path(source) # folder to take images from
    pth_proc=Path(destination) # folder to save processed images
    
    list_images=list(pth.glob("*.*"))
    for image in list_images[:10]:
        path=image
        path_str=image.as_posix()
        img=cv2.imread(path_str, cv2.IMREAD_COLOR) # reading image in       
        faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5) # applying haarcascade classifier to all raw images, detecting face coordinates on the image
        if type(faces_detected)==tuple: # if face is not recognised jump to the next image
            continue
        else:
            if faces_detected.shape[0]>1: # if more than 1 face is recognized jump to the next image
                continue
            else:    
                (x, y, w, h) = faces_detected[0] # coordinates of face on the image
                while y<y_c:
                    y_c=y_c-5
                while x<x_c:
                    x_c=x_c-5
                pth_save=pth_proc / path.parts[-1]
                # cropping image, saving extrated faces to separate folder
                cv2.imwrite(pth_save.as_posix(), img[y-y_c+1:y+h+y_c, x-x_c+1:x+w+x_c])
                k=k+1
                
#%%
source_mask=r'C:\Users\aermolov\Documents\ML_data\probearbeit_masks\covered'
source_nomask=r'C:\Users\aermolov\Documents\ML_data\probearbeit_masks\bare'
def resize(source_mask, source_nomask, width, height): ### Function to resize pictures and create labels
    X=[]
    y=[]
    new_dim=(width, height)
    i=0
    pth_maske=Path(source_mask)
    pth_nomaske=Path(source_nomask)
    list_images=list(pth_maske.glob("*.*"))+list(pth_nomaske.glob("*.*")) # collect all image locations to one list
    random.shuffle(list_images) # shuffle images in the list
    for image in list_images:   
        path_str=image.as_posix()
        temp=cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
        X.append(cv2.resize(temp, new_dim, interpolation=cv2.INTER_LINEAR)) # resize image to 160x160 pixels
        if "bare" in path_str: # if image is from "no mask" folder, label as 0
            y.append(0)
        if "covered" in path_str: # if image is from "mask" folder, label as 1
            y.append(1)
        i=i+1
        print("Performed: {0} %".format(round(i/len(list_images)*100, 2)))
    X=np.array(X)
    y=np.array(y)
    np.save("inputs", X) # save images in one multidimensional numpy array
    np.save('labels', y) # save labels to numpy array
    return X, y
    
#%%
X=np.load(r'C:\Users\aermolov\Google Drive\portfolio\mask_detection\inputs.npy') # upload saved images
y=np.load(r'C:\Users\aermolov\Google Drive\portfolio\mask_detection\labels.npy') # upload saved labels

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.2, random_state=0) # 20% of images as a validation set
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255, # applying pixel normalization and data augmentation to training set
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255) # applying pixel normalization to validation set
train_gen=train_datagen.flow(X_train, y_train, batch_size=32) # final training set generation with batch size of 32
val_gen=val_datagen.flow(X_val, y_val, batch_size=32) # final validation set generation with batch size of 32

import keras # importing keras modules build classification model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

#%%
### model structure
model=Sequential()
model.add(Conv2D(filters=32, kernel_size = (3,3), activation='relu', input_shape=(160,160,3)))
#model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size = (3,3), activation='relu'))
#model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    
model.add(MaxPooling2D(pool_size=(2,2)))
    
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(1,activation="sigmoid")) # In the last dense layer sigmoid activation is used because model has binary output 
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.RMSprop(lr=1e-3), metrics=["accuracy"])

#%%
history=model.fit(train_gen, epochs=40, validation_data=val_gen) # training the model

#%%
# Plot loss and accuracy
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs=range(1, len(acc)+1)
from matplotlib import pyplot as plt
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs,val_acc, 'r', label='Validation accuracy')
plt.legend()
plt.show()