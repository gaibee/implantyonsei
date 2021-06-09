from keras import layers, models
from keras.models import load_model
from keras.applications import InceptionV3, Xception
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2
from keras.utils.vis_utils import plot_model
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math
import os
import pickle


def train_model(p_model):

    print('[INFO] building model...') #base 모델을 만듦
    pic_size = None
    base_model = None
    
    if(p_model.lower() == 'inceptionv3'):
        base_model = InceptionV3(
            include_top=False,
            input_shape=(299, 299, 3))
        pic_size = (299, 299)
    elif(p_model.lower()=='xception'):
        base_model = Xception(
            include_top=False,
            input_shape=(229, 229, 3))
        pic_size = (229, 229)
    elif(p_model.lower()=='resnet50v2'):
        base_model = ResNet50V2(
            include_top=False,
            input_shape=(224, 224, 3))
        pic_size = (224, 224)
    elif(p_model.lower()=='resnet101v2'):
        base_model = ResNet101V2(
            include_top=False,
            input_shape=(224, 224, 3))
        pic_size = (224, 224)
    else:
        print('존재하지 않는 모델입니다 : {}'.format(p_model))
        return

    # 모델을 만들고
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    output_tensor = layers.Dense(3, activation='softmax')(x)

    # 모델 정의
    model = Model(inputs=base_model.input, outputs=output_tensor)
    
    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

    print('[INFO] data generator begin...')
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       vertical_flip=True,
                                       rotation_range=10,
                                       width_shift_range=0.05,
                                       height_shift_range=0.05,
                                       shear_range=0.1)

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    
    for i in range(10):
        train_dir = os.path.join('dataset/train_splited/kfold', str(i), 'train')
        test_dir = os.path.join('dataset/train_splited/kfold', str(i), 'test')
        
        train_generator = train_datagen.flow_from_directory(train_dir, batch_size=16, target_size=pic_size, color_mode='rgb')
        test_generator = train_datagen.flow_from_directory(test_dir, batch_size=16, target_size=pic_size, color_mode='rgb')
        
        checkpoint = ModelCheckpoint(filepath='model/{}/checkpoint_{}.hdf5'.format(i, p_model),
                                     monitor='loss',
                                     mode='auto',
                                     save_best_only=True)

        print('[INFO] learning begin for k-fold {}...'.format(i))
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=math.ceil(train_generator.n / train_generator.batch_size),
                                      epochs=100,
                                      validation_data=test_generator,
                                      validation_steps=math.ceil(test_generator.n / test_generator.batch_size),
                                      callbacks=[checkpoint])

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure()
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Test acc')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig('model/{}/{}_fig_Accuracy.jpg'.format(i, p_model))

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Test loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig('model/{}/{}_fig_Loss.jpg'.format(i, p_model))

        try:
            with open(r'model/{}/{}_history.pkl'.format(i, p_model), 'wb') as f:
                pickle.dump(history, f)
        except Exception as e:
            return
        
train_model('xception')