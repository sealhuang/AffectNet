import os, h5py
import scipy.io as mat_load
import numpy as np
np.random.seed(1337)  # for reproducibility

import numpy.random as rng

# TODO: check on the size of input images MOON paper
img_width, img_height, depth = 224,224,3
nb_train_samples = 297803
#nb_train_samples = 1000
nb_validation_samples = 4500
#nb_validation_samples = 1200
nb_epoch = 50
batch_size = 64

def val_generator():
    #full_image_dir = 'keras_data/full_image/train'
    img_dir = '../data_affect/val'
    label_file = mat_load.loadmat('landmark_val_labels')
    label_data = label_file['val_label']
    #landmark_data = label_file['landmark_labels']
    train_batch_size = batch_size
    image_id = 1
    val_index_thr = batch_size * int(nb_validation_samples/batch_size)

    while True:
        batch_labels = []
        batch_landmark_labels = []
        batch_feature = []
        image_count = 0

        if((image_id+batch_size) > val_index_thr):
            image_id = 1

        while(image_count < batch_size and image_id < nb_validation_samples):
            try:

                img_label = label_data[image_id-1][0]
                landmark_label = label_data[image_id-1][1:]
                batch_labels += [img_label]
                image_count = image_count + 1
                image_id = image_id + 1
            except IOError:
                image_id = image_id + 1

                continue

        #batch_labels = to_categorical(batch_labels,9)
        batch_labels = np.array(batch_labels)

        yield (batch_labels)


[batch_labels] = next(val_generator())
print(batch_labels.shape)

for i in range(64):
    print(batch_labels[i])
