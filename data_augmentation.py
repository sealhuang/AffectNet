from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

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
count = 1
img_dir = '../data_affect/train/'
no_aug_images = 5

while(count < no_images):
    filename = img_dir + '/image' + str(count).zfill(7) + '.jpg'
    prefix_str = 'image_' + str(count)
    img = load_img(filename)
    x = img_to_array(img)
    x1 = []
    x1 = x.reshape((1,) + x.shape)
    count = count + 1
    l = [1]
    label = np.array(l)
    print(x1.shape)
    print(label.shape)

    i = 0
    for x_batch, y_batch in datagen.flow(x1, y=label,batch_size=1, save_to_dir='data_augment', save_prefix= prefix_str, save_format='jpeg'):
        i+= 1
        if i > 5:
            break
