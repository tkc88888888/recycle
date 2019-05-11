# import the necessary packages
import tensorflow as tf
import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
keras.backend.set_session(sess)




from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model


from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input


def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))

	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))

	# return our model
	return model

def create_cnn(dropout, fc_layers, num_classes, regress=False):


	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	HEIGHT = 224
	WIDTH = 224
	
	base_model = MobileNetV2(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3))
	

    	for layer in base_model.layers:
            layer.trainable = False

	x = base_model.output
        x = Flatten()(x)
	for fc in fc_layers:
	    # New FC layer, random init
	    x = Dense(fc, activation='relu')(x) 
	    x = Dropout(dropout)(x)



	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(4,activation='relu')(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="linear")(x)

    	model = Model(inputs=base_model.input, outputs=x)
	# return the CNN
	
	return model
