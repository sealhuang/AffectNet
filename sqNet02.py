import os, h5py
import scipy.io as mat_load
import numpy as np
np.random.seed(1337)  # for reproducibility

import numpy.random as rng
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import optimizers,regularizers
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D,concatenate,GlobalAveragePooling2D,AveragePooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense,Input,BatchNormalization
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import  image
from keras.applications.vgg16 import preprocess_input
from keras.applications import vgg16, vgg19,inception_v3
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

img_width, img_height, depth = 227,227,3
img_input = Input(shape=(img_height,img_width,depth))
nb_train_samples = 37553
nb_validation_samples = 4000
nb_epoch = 50
batch_size = 64
run_id = 2

# data generator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, \
        zoom_range=0.2, horizontal_flip=True)

#train_datagen = ImageDataGenerator(shear_range=0.2, \
#        zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '../class_dir/train_class',  # this is the target directory
        target_size=(227, 227),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../class_dir/val_class',
        target_size=(227,227),
        batch_size=batch_size,
        class_mode='categorical')

# define the squeeze-model
def fire_module(fire_input, fire_id, squeeze, expand):
    s_id = 'fire' + str(fire_id) + '/'

    x      = Conv2D(squeeze, (1, 1), padding='same', kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name=s_id + 'sq1x1')(fire_input)
    x      = BatchNormalization()(x)

    tower01 = Conv2D(expand, (1, 1), padding='same', kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name=s_id + 'exp1x1')(x)
    tower01 = Conv2D(expand, (3, 3), padding='same', kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name=s_id + 'exp3x3')(tower01)
    tower01 = Dropout(0.2)(tower01)

    tower02 = Conv2DTranspose(expand, (1, 1), padding='same', kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name=s_id + 'd_exp1x1')(x)
    tower02 = Conv2DTranspose(expand, (3, 3), padding='same', kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name=s_id + 'd_exp3x3')(tower02)
    tower02 = Dropout(0.2)(tower02)

    y      = concatenate([tower01, tower02], axis=3, name=s_id + 'concat')
    return y

#Build the network
x = Conv2D(96, (7, 7), strides = (2,2), kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name='block1_conv1')(img_input)
x = MaxPooling2D((3, 3), strides=(2, 2), name='pool01')(x)
#x = BatchNormalization()(x)

x = fire_module(x, fire_id=2, squeeze=16, expand=64)
x = fire_module(x, fire_id=3, squeeze=16, expand=64)
x = fire_module(x, fire_id=4, squeeze=32, expand=128)
x = MaxPooling2D((3, 3), strides=(2, 2), name='pool02')(x)
#x = BatchNormalization()(x)

x = fire_module(x, fire_id=5, squeeze=32, expand=128)
#x = fire_module(x, fire_id=10, squeeze=32, expand=128)
x = fire_module(x, fire_id=6, squeeze=48, expand=192)
#x = fire_module(x, fire_id=11, squeeze=48, expand=192)
#x = MaxPooling2D((3, 3), strides=(2, 2), name='pool04')(x)
x = fire_module(x, fire_id=7, squeeze=48, expand=192)
x = fire_module(x, fire_id=8, squeeze=64, expand=256)
#x = fire_module(x, fire_id=12, squeeze=64, expand=256)
x = MaxPooling2D((3, 3), strides=(2, 2), name='pool03')(x)
x = fire_module(x, fire_id=9, squeeze=64, expand=256)
x = Dropout(0.5)(x)

#x = Conv2D(64, (3, 3), strides = (1,1), kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name='conv11')(x)
x = Conv2D(8, (1, 1), padding='same', kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer = regularizers.l2(0.0001),data_format='channels_last', activation= 'relu',name='conv10')(x)
x = GlobalAveragePooling2D()(x)
#x = AveragePooling2D((13,13),name='avg_pool001')(x)
#flatten = Flatten(name='flatten01')(x)
predictions = Activation('softmax',name='softmax001')(x)

final_model = Model(inputs = img_input,outputs = predictions)
final_model.summary()

optimizer_adam = optimizers.Adam(lr = 0.0001)
final_model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_adam,
              metrics=['accuracy'])

# checkpoint
outputFolder = './output-model-scratch-sqnet02'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/weights_l0.0001_d0.02-{epoch:03d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, \
                             save_best_only=False, save_weights_only=True, \
                             mode='auto', period=1)
callbacks_list = [checkpoint]

#TODO: train_generators and validation train_generators
#TODO: train labels and validation labels

print('running '+ str(run_id))
n_steps_per_epoch = nb_train_samples/batch_size
n_val_steps = nb_validation_samples/batch_size
final_model.fit_generator(generator=train_generator,steps_per_epoch=n_steps_per_epoch,callbacks=callbacks_list, \
                          epochs=nb_epoch,validation_data = validation_generator,validation_steps=n_val_steps)

final_model_json = final_model.to_json()
with open("sqNet02_d0.02_json.json", "w") as json_file:
    json_file.write(final_model_json)
