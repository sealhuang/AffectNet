import os, h5py
import scipy.io as mat_load
import numpy as np
np.random.seed(1337)  # for reproducibility

import numpy.random as rng
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import optimizers,regularizers
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D,concatenate,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input,BatchNormalization
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import  image
from keras.applications.vgg16 import preprocess_input
from keras.applications import vgg16, vgg19,inception_v3
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

img_width, img_height, depth = 224,224,3
img_input = Input(shape=(img_height,img_width,depth))
nb_train_samples = 37553
nb_validation_samples = 4000
nb_epoch = 50
batch_size = 64

# data generator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, \
        zoom_range=0.2, horizontal_flip=True)

#train_datagen = ImageDataGenerator(shear_range=0.2, \
#        zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '../class_dir/train_class',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../class_dir/val_class',
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical')

# Build the model
# Block 1
x = Conv2D(96, (11, 11), strides = (4,4), kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name='block1_conv1')(img_input)
x = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool')(x)
x = BatchNormalization()(x)

x = Conv2D(256, (5,5), activation= 'relu',kernel_initializer='random_uniform',bias_initializer='zeros',name='block2_conv1')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool')(x)
x = BatchNormalization()(x)

x = Conv2D(384, (3, 3), strides = (1,1), activation= 'relu', padding='same', kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),name='block3_conv1')(x)
x = Conv2D(384, (3, 3), strides = (1,1), activation= 'relu', padding='same', kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),name='block3_conv2')(x)
x = Conv2D(256, (3, 3), strides = (1,1), activation= 'relu', padding='same', kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),name='block3_conv3')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001), name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),name='fc4')(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='softmax', kernel_initializer='random_uniform',bias_initializer='zeros',name='fc3')(x)

final_model = Model(inputs = img_input,outputs = predictions)

optimizer_adam = optimizers.Adam(lr = 0.0001)
final_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_adam,
              metrics=['accuracy'])

# checkpoint
outputFolder = './output-model-scratch'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/weights_t1024_t512-{epoch:03d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, \
                             save_best_only=False, save_weights_only=True, \
                             mode='auto', period=1)
callbacks_list = [checkpoint]

#TODO: train_generators and validation train_generators
#TODO: train labels and validation labels


n_steps_per_epoch = nb_train_samples/batch_size
n_val_steps = nb_validation_samples/batch_size
final_model.fit_generator(generator=train_generator,steps_per_epoch=n_steps_per_epoch,callbacks=callbacks_list, \
                          epochs=nb_epoch,validation_data = validation_generator,validation_steps=n_val_steps)

final_model_json = final_model.to_json()
with open("from_scratch_top_layer_1024_1024_512_dropout_0.5_json.json", "w") as json_file:
    json_file.write(final_model_json)
