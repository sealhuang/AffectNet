from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from os import listdir,remove
import shutil
import os

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
no_images = 27000
img_count = 22001
dst_img_count = 131971
img_dir = '../data_affect/train'
no_aug_images = 6
aug_images_list = []

while(img_count <= no_images):
    filename = img_dir + '/image' + str(img_count).zfill(7) + '.jpg'
    print(filename)
    prefix_str = 'image_' + str(img_count)
    img = load_img(filename)
    x = img_to_array(img)
    x1 = []
    x1 = x.reshape((1,) + x.shape)
    img_count = img_count + 1
    l = [1]
    label = np.array(l)

    i = 0
    os.mkdir(parent_folder)
    for x_batch, y_batch in datagen.flow(x1, y=label,batch_size=1, save_to_dir=parent_folder, save_prefix= prefix_str, save_format='jpeg'):
        i+= 1
        if i > (no_aug_images-1):
            break

    files = listdir(parent_folder)
    files_count = 1

    for name in files:
        dst_filename = destination_folder + 'image' + str(dst_img_count).zfill(7) + '.jpeg'
        print(dst_filename)
        src_filename = parent_folder + str(name)
        shutil.copyfile(src_filename,dst_filename)
        remove(src_filename)
        dst_img_count = dst_img_count + 1
        files_count = files_count + 1

    files_count -= 1
    aug_images_list += [files_count]

    os.rmdir(parent_folder)

thefile = open('aug_list.txt', 'a+')
for item in aug_images_list:
  thefile.write("%s\n" % item)

thefile.close()
