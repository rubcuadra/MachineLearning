#https://github.com/0bserver07/Keras-SegNet-Basic
from __future__ import absolute_import
from __future__ import print_function
import os
# os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'
import keras.models as models
from keras.layers import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from camvid_loader import get_data
import cv2
import numpy as np
import json
import seaborn as sns
from scipy.misc import imsave

def saveToImages(data_path, predictions): #Saves the matrix as image with different color for each tag
    num_classes = 12
    width, height = 360, 480
    color_palette = sns.color_palette("hls", num_classes)

    for i,p in enumerate(predictions):
        img = np.zeros( (width,height,3) ) #3 rgb
        matrix_tags = np.argmax( p.reshape((width,height,num_classes)), axis=2)
        for ii in range(width): #Slow
            for jj in range(height):
                img[ii][jj] += color_palette[ matrix_tags[ii][jj] ]     
        imsave(f"{data_path}{i}.png",img) 



if __name__ == '__main__':
    #You should run create_model.py first
    K.set_image_dim_ordering('th') #We dont have gpu but created the model with gpu format
    np.random.seed(7) # Controlled seed

    data_shape = 360*480
    #good init for weights
    class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

    # process the data
    train_data, train_label, test_data, test_label, val_data, val_label = get_data()
    # or load it if we saved it before, those files are really heavy
    # train_data = np.load('./data/train_data.npy')
    # train_label = np.load('./data/train_label.npy')
    # test_data = np.load('./data/test_data.npy')
    # test_label = np.load('./data/test_label.npy')

    # load the model created with the script create_model.py
    with open('model/segNet_model.json') as model_file: segnet = models.model_from_json(model_file.read())
    segnet.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    #Params & Checkpoints
    checkpoint = ModelCheckpoint("weights/weights.best.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    nb_epoch = 100
    batch_size = 6

    if True: #TRAIN AND SAVE
        # Fit the model
        history = segnet.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
                            verbose=1, class_weight=class_weighting , validation_data=(test_data, test_label), shuffle=True) # validation_split=0.33
        segnet.save_weights('weights/model_{}.hdf5'.format(nb_epoch))
    else: #PREDICT, takes some time
        segnet.load_weights('weights/weights_best.hdf5') #After training you should put the name of the weights file
        pred = segnet.predict( test_data )
        saveToImages("predictions/", pred)
        # np.save("preds", pred)
        score = segnet.evaluate(test_data, test_label, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        # Test loss: 0.9960364162154464
        # Test accuracy: 0.6515996364053227
