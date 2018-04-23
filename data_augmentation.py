from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

no_images = 2
count = 1
img_dir = '../data_affect/train/'

while(count < no_images):
    filename = img_dir + '/image' + str(image_id).zfill(7) + '.jpg'
    img = load_img(filename)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, y=1,batch_size=1, save_to_dir='data_augment', save_prefix='image', save_format='jpeg'):
        i += 1
        if i > 10:
            break  # otherwise the generator would loop indefinitely
