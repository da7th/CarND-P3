import os
import numpy as np
import csv
from scipy import misc
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

folder = "data/"
image_files = os.listdir(folder)
image_size_h = 160
image_size_w = 320

csv_field_names = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
center_images = []
left_images = []
right_images = []
steering_angles = []
with open("data/driving_log.csv") as csvfile:
    # reader = csv.DictReader(csvfile, fieldnames = csv_field_names)
    # for row in reader:
    #     center_images.append(row['center_image'])
    #     left_images.append(row['left_image'])
    #     right_images.append(row['right_image'])
    #     steering_angles.append((float(row['steering_angle'])+1)*90)
    reader = csv.DictReader(csvfile)
    for row in reader:
        center_images.append(row['center'])
        left_images.append(row['left'])
        right_images.append(row['right'])
        steering_angles.append((float(row['steering'])+1)*90)
print(len(center_images))

center_image_dataset = []
left_image_dataset = []
right_image_dataset = []

for img in center_images:
    if img[:4] == "IMG/":
        center_image_dataset.append(misc.imresize(misc.imread("data/" + img), size=[32, 32, 3]))
    else:
        center_image_dataset.append(misc.imresize(misc.imread(img[41:]), size=[32,32,3]))
for img in left_images:
    if img[:4] == " IMG":
        left_image_dataset.append(misc.imresize(misc.imread("data/" + img[1:]), size=[32,32,3]))
    else:
        left_image_dataset.append(misc.imresize(misc.imread(img[42:]), size=[32,32,3]))
for img in right_images:
    if img[:4] == " IMG":
        right_image_dataset.append(misc.imresize(misc.imread("data/" + img[1:]), size=[32,32,3]))
    else:
        right_image_dataset.append(misc.imresize(misc.imread(img[42:]), size=[32,32,3]))

center_image_dataset = np.asarray(center_image_dataset)
print("center shape:", center_image_dataset.shape)
left_image_dataset = np.asarray(left_image_dataset)
right_image_dataset = np.asarray(right_image_dataset)

print("center_image_dataset:", len(center_image_dataset),\
"\nleft_image_dataset:", len(left_image_dataset),\
"\nright_image_dataset:", len(right_image_dataset),\
"\nwith the shape:", center_image_dataset[0].shape)

dataset_inputs = np.concatenate((center_image_dataset, left_image_dataset, right_image_dataset), axis=0)
dataset_labels = np.concatenate((steering_angles, steering_angles, steering_angles), axis=0)

print("label example:", dataset_labels[100])
print("inputs shape:",dataset_inputs.shape, \
"\nlabels shape:", dataset_labels.shape, \
"\nlabels type:", dataset_labels[0].dtype)

dataset_inputs, dataset_labels = shuffle(dataset_inputs, dataset_labels)
dataset_inputs = dataset_inputs
dataset_labels = dataset_labels

uniques = np.unique(dataset_labels)
uniques = len(uniques) + 1
print("unique cases:", uniques)

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Activation('relu'))
model.add(Dropout(0.50))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1))

# TODO: Compile and train the model

model.compile(loss='mse', optimizer='adam')


history = model.fit(dataset_inputs, dataset_labels, batch_size=512, nb_epoch=40, validation_split=0.2)


model.save("model.h5")



#END
