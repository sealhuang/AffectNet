from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from os import listdir,remove
import shutil

parent_folder = 'buffer/'
destination_folder = 'data_augment/'

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

nb_train_samples = 42553
no_images = 10
img_count = 1
dst_img_count = 1
img_dir = '../data_affect/train/'
no_aug_images = 5

while(img_count < no_images):
    filename = img_dir + '/image' + str(img_count).zfill(7) + '.jpg'
    prefix_str = 'image_' + str(img_count)
    img = load_img(filename)
    x = img_to_array(img)
    x1 = []
    x1 = x.reshape((1,) + x.shape)
    img_count = img_count + 1
    l = [1]
    label = np.array(l)
    print(x1.shape)
    print(label.shape)

    i = 0
    os.mkdir(parent_folder)
    for x_batch, y_batch in datagen.flow(x1, y=label,batch_size=1, save_to_dir='parent_folder', save_prefix= prefix_str, save_format='jpeg'):
        i+= 1
        if i > 5:
            break

    files = listdir(parent_folder)
    files_count = 1

    for name in files:
        dst_filename = destination_folder + 'image' + str(dst_img_count).zfill(7) + '.jpeg'
        src_filename = parent_folder + str(name)
        shutil.copyfile(src_filename,dst_filename)
        remove(src_filename)
        dst_img_count = dst_img_count + 1

    os.rmdir(parent_folder)
