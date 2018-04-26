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


def val_generator():
    #full_image_dir = 'keras_data/full_image/train'
    img_dir = '../data_affect/val'
    label_file = mat_load.loadmat('landmark_val_labels')
    label_data = label_file['val_label']
    #landmark_data = label_file['landmark_labels']
    train_batch_size = batch_size
    image_id = 1
    train_index_thr = batch_size * int(nb_validation_samples/batch_size)

    while True:
        batch_labels = []
        batch_landmark_labels = []
        batch_feature = []
        image_count = 0

        if((image_id+batch_size) > train_index_thr):
            image_id = 1

        while(image_count < batch_size and image_id < nb_validation_samples):
            try:
                filename = img_dir + '/image' + str(image_id).zfill(7) + '.jpg'
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


# Set-up the architecture
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
#weights_path = 'keras_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Block 1
x = Conv2D(64, (3, 3), data_format='channels_last',padding = 'same', activation= 'relu',name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation= 'relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

vgg16_model = Model(img_input,x)
weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
vgg16_model.load_weights(weights_path)

for layers in vgg16_model.layers:
    layers.trainable = False

data_shape = vgg16_model.output_shape[1:]
output_from_vgg16_model = Input(shape = (data_shape))

x = Flatten(name='flatten')(output_from_vgg16_model)
x = Dense(512, activation='relu', name='t1_fc3')(x)
#x = Dropout(0.6)(x)
x = Dense(512, activation='relu', name='t1_fc1')(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu', name='t1_fc2')(x)
predictions = Dense(9, activation='softmax', name='predictions')(x)

discrete_top_model = Model(inputs = output_from_vgg16_model, outputs = predictions)

x = Flatten(name='flatten')(output_from_vgg16_model)
#x = Dense(1024, activation='relu', name='fc3')(x)
x = Dense(512, activation='relu', name='t2_fc1')(x)
x = Dropout(0.6)(x)
x = Dense(128, activation='relu', name='t2_fc2')(x)
continuous_prediction = Dense(2, activation='tanh', name='continuous_prediction')(x)

coninuous_top_model = Model(inputs = output_from_vgg16_model, outputs = continuous_prediction)

final_model = Model(inputs = vgg16_model.input,outputs = [discrete_top_model(vgg16_model.output),coninuous_top_model(vgg16_model.output)])

# TODO Step 3: compiling and training
optimizer_adam = optimizers.Adam(lr = 0.0001)
optimizer_rmsprop = optimizers.RMSprop(lr=0.00001)
final_model.compile(loss = ['categorical_crossentropy','mean_squared_error'], \
                     optimizer = optimizer_adam, metrics=['accuracy'])


# checkpoint
outputFolder = './output-affect'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/weights_c_t1024_t512_t256-{epoch:03d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                             save_best_only=False, save_weights_only=True, \
                             mode='auto', period=1)
callbacks_list = [checkpoint]

n_steps_per_epoch = nb_train_samples/batch_size
n_val_steps = nb_validation_samples/batch_size
final_model.fit_generator(generator=train_generator(),steps_per_epoch=n_steps_per_epoch,callbacks=callbacks_list, \
                          epochs=nb_epoch,validation_data = val_generator(),validation_steps=n_val_steps, \
                          initial_epoch=0)
final_model.save_weights('mulitask01_e50_t512_512_256.h5')

final_model_json = final_model.to_json()
with open("vgg16_top_layer_512_512_256_json.json", "w") as json_file:
    json_file.write(final_model_json)
