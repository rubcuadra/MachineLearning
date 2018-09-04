import numpy as np
import keras
from pickle import load, dump
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
from random import seed,sample,random
seed(316)

numExamplesToUse = 0 #1500 was working ok #TOTAL: 13785

#Hyperparams

batch_size = 128 #Examples Per Iter GDA
epochs = 15      #Iters over examples
#Kernel Sizes = (3,3) #in Convul layer
#Num Kernels per Convul Layer
#Dropout Probability
#Pools Sizes (2,2)
#Num Neurons in ML Layer
num_classes = 3 #Obtainable using np.unique(tags)

if __name__ == '__main__':
    
    #Read files
    with open("vehicleimg.dms","rb")   as dp: images    = load( dp )
    with open("vehicletrgt.dms","rb")  as dp: tags      = load( dp )
    
    #Load just some data
    if numExamplesToUse:
        ixs = sample(range(0, len(tags)), numExamplesToUse)
        images = np.array( [images[i] for i in ixs] )
        tags   = np.array( [tags  [i] for i in ixs] )
    
    images = images.astype('float32')
    
    tags = np.vectorize(lambda t: t-2)(tags) #Starting from 0
    tags = to_categorical(tags, num_classes) #Matrix of N,num_classes
    
    x_train, x_test, y_train, y_test = train_test_split(images,tags,test_size=0.2,random_state=0) #CrossVal

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=images[0].shape, activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25)) 

    model.add(Conv2D(32, (3, 3),activation="relu"))
    model.add(Conv2D(32, (3, 3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25)) 

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) 

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

    # model = keras.models.load_model( 'first_try.hdf5' )
    # plot_model(m, to_file='first_try.png')
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(f'{random()}.hdf5')
    
    
    # batch_size = 128
    # num_classes = 10
    # epochs = 3

    # # input image dimensions
    # img_rows, img_cols = 28, 28

    # # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # if K.image_data_format() == 'channels_first':
    #     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    #     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    #     input_shape = (1, img_rows, img_cols)
    # else:
    #     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    #     input_shape = (img_rows, img_cols, 1)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])

    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(x_test, y_test))
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # model.save(f'mnist{random()}.hdf5')

    

