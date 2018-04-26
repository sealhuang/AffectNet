import os, h5py
import scipy.io as mat_load
import numpy as np
np.random.seed(1337)  # for reproducibility

import numpy.random as rng
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import optimizers
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D,concatenate
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import  image
from keras.applications.vgg16 import preprocess_input
from keras.applications import vgg16, vgg19,inception_v3
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

# TODO: check on the size of input images MOON paper
img_width, img_height, depth = 224,224,3
img_input = Input(shape=(img_height,img_width,depth))
nb_train_samples = 255250
#nb_train_samples = 1000
nb_validation_samples = 4500
#nb_validation_samples = 1200
nb_epoch = 50
batch_size = 64

#Step 1: Train generator
def train_generator():
    #full_image_dir = 'keras_data/full_image/train'
    img_dir = '../augmented_images'
    label_file = mat_load.loadmat('da_train_labels_5000')
    label_data = label_file['da_train_labels']
    #landmark_data = label_file['landmark_labels']
    train_batch_size = batch_size
    image_id = 1
    train_index_thr = batch_size * int(nb_train_samples/batch_size)

    while True:
        batch_labels = []
        batch_landmark_labels = []
        batch_feature = []
        image_count = 0

        if((image_id+batch_size) > train_index_thr):
            image_id = 1


        while(image_count < batch_size and image_id < nb_train_samples):
            try:
                filename = img_dir + '/image' + str(image_id).zfill(7) + '.jpeg'
                body_img = image.load_img(filename, target_size=(img_height,img_width))
                x = image.img_to_array(body_img)
                x = np.expand_dims(x, axis=0)
                feature01 = preprocess_input(x)
                img_label = label_data[image_id-1][0]

                landmark_label = label_data[image_id-1][1:]
                batch_feature += [feature01]
                batch_labels += [img_label]
                batch_landmark_labels += [landmark_label]
                image_count = image_count + 1
                image_id = image_id + 1
            except IOError:
                image_id = image_id + 1
                continue

        batch_labels = to_categorical(batch_labels,9)
        batch_labels = np.array(batch_labels)
        batch_landmark_labels = np.array(batch_landmark_labels)
        batch_feature = np.array(batch_feature)
        batch_feature = np.squeeze(batch_feature)

        yield (batch_feature,[batch_labels,batch_landmark_labels])

[feature01,[labels01,c_labels02]] = next(train_generator())
print(feature01.shape)
print(labels01.shape)
print(c_labels02.shape)
