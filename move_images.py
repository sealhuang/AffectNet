import numpy as np
from os import listdir,remove
import shutil

train_dir = '../data_affect/train'
aug_dir = '../augmented_images'
data_train_aug_dir = '../data_affect/train_aug'

count = 1
no_train_images = 42553
no_augmented_images = 255250


for i in range(no_train_images):
    image_id = i+1
    src_filename = train_dir + '/image' + str(image_id).zfill(7) + '.jpg'
    dst_filename = data_train_aug_dir + '/image' + str(count).zfill(7) + '.jpg'
    shutil.copyfile(src_filename,dst_filename)
    count = count + 1

for i in range(no_augmented_images):
    image_id = i + 1
    src_filename = aug_dir + '/image' + str(image_id).zfill(7) + '.jpeg'
    dst_filename = data_train_aug_dir + '/image' + str(count).zfill(7) + '.jpeg'
    shutil.copyfile(src_filename,dst_filename)
    count = count + 1
