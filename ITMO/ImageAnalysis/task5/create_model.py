from __future__ import absolute_import
from __future__ import print_function
import os
# Install the plaidml backend
# import plaidml.keras
# plaidml.keras.install_backend()

import keras.models as models
from keras.layers import Layer, Dropout, Activation, Reshape, Permute, Conv2D , MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import json
K.set_image_dim_ordering('th')
np.random.seed(7) 



#HParams
kernel = 3
filter_size = 64
pad = 1
pool_size = 2

#Start Segnet
segnet = models.Sequential()
segnet.add(Layer(input_shape=(3, 360, 480))) #RGB, Width, Height -> thanos format

#Encoding Layers
segnet.add( ZeroPadding2D(padding=(pad,pad)) )
segnet.add( Conv2D(filter_size, (kernel, kernel), padding="valid") )
segnet.add( BatchNormalization() )
segnet.add( Activation('relu') )
segnet.add( MaxPooling2D(pool_size=(pool_size, pool_size)) )

segnet.add( ZeroPadding2D(padding=(pad,pad)) )
segnet.add( Conv2D(128, (kernel, kernel), padding="valid") )
segnet.add( BatchNormalization() )
segnet.add( Activation('relu') )
segnet.add( MaxPooling2D(pool_size=(pool_size, pool_size)) )

segnet.add( ZeroPadding2D(padding=(pad,pad)) )
segnet.add( Conv2D(256, (kernel, kernel), padding="valid") )
segnet.add( BatchNormalization() )
segnet.add( Activation('relu') )
segnet.add( MaxPooling2D(pool_size=(pool_size, pool_size)) )

segnet.add( ZeroPadding2D(padding=(pad,pad)) )
segnet.add( Conv2D(512, (kernel, kernel), padding="valid") )
segnet.add( BatchNormalization() )
segnet.add( Activation('relu') )

#Decoding Layers
segnet.add( ZeroPadding2D(padding=(pad,pad)) )
segnet.add( Conv2D(512, (kernel, kernel), padding="valid") )
segnet.add( BatchNormalization() )

segnet.add( UpSampling2D(size=(pool_size,pool_size)) )
segnet.add( ZeroPadding2D(padding=(pad,pad)) )
segnet.add( Conv2D(256, (kernel, kernel), padding="valid") )
segnet.add( BatchNormalization() )

segnet.add( UpSampling2D(size=(pool_size,pool_size)) )
segnet.add( ZeroPadding2D(padding=(pad,pad)) )
segnet.add( Conv2D(128, (kernel, kernel), padding="valid") )
segnet.add( BatchNormalization() )

segnet.add( UpSampling2D(size=(pool_size,pool_size)) )
segnet.add( ZeroPadding2D(padding=(pad,pad)) )
segnet.add( Conv2D(filter_size, (kernel, kernel), padding="valid") )
segnet.add( BatchNormalization() )

#Final Layer
segnet.add( Conv2D(12, (1,1), padding="valid") )
segnet.add( Reshape((12, 360*480), input_shape=(12,360,480))) #12 classes; 360 x 480 pixels per Image
segnet.add( Permute((2, 1)))
segnet.add( Activation('softmax') )

#Print and save network architecture
if __name__ == '__main__': #If they import it ok, if run it then save
	segnet.summary()
	plot_model(segnet, to_file="model/segNet_model.png")
	with open('model/segNet_model.json', 'w+') as outfile: outfile.write(json.dumps(json.loads(segnet.to_json()), indent=2))

