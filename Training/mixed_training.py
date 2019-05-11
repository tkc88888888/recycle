#modified from https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/


# USAGE
# python mixed_training.py --dataset Wastes\ Dataset/


import tensorflow as tf
import keras

#CUDNN GPU error remedy. without this will face CUDNN_STATUS_ALLOC_FAILED
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
keras.backend.set_session(sess)




# import the necessary packages
import datasets
import models

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model

import numpy as np
import argparse
import locale
import os
################################################################################# PREPROCESS DATA GOT FROM DATASETS.PY 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of waste images")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each waste in the dataset and then load the dataset
print("[INFO] loading waste attributes...")
inputPath = os.path.sep.join([args["dataset"], "WastesInfo.txt"])
df = datasets.load_waste_attributes(inputPath)

# load the waste images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading waste images...")
(images,labels) = datasets.load_waste_images_labels(df, args["dataset"])
images = np.array(images, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(df, images, labels, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX, trainY, testY) = split

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=6)
testY = to_categorical(testY, num_classes=6)

# process the waste attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
(trainAttrX, testAttrX) = datasets.process_waste_attributes(df,
	trainAttrX, testAttrX)



################################################################################# INITIALIZE VARIABLES

#"""
# INITIALIZE the model if training start here
print("[INFO] compiling model...")

chanDim = -1
class_list = ["aluminium", "cardboard", "glass", "paper", "plastic", "thrash"]
FC_LAYERS = [1024, 1024]
dropout = 0.5
EPOCHS= 8 #200
BS = 8
INIT_LR = 1e-3
#INIT_LR/EPOCHS = (1e-3)/8
HEIGHT = 224
WIDTH = 224


######################################################################################################### CHOOSE TRAINING OR RETRAINING WITH AND MODELS.PY

#(A) CREATE the MLP and CNN models							#CHOOSE EITHER A OR B
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(dropout, FC_LAYERS, len(class_list), regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(8, activation="relu")(combinedInput)

# New softmax layer
x = Dense(len(class_list), activation='softmax')(x) 
    
# our final model will accept numerical data on the MLP
# input and images on the CNN input, outputting a predicted category of waste)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

"""#########################################################

#(B) LOAD weights if not training start here(shortcut)					#CHOOSE EITHER A OR B
#model = model.load_weights("./checkpoints/weights.best.hdf5")
model = load_model("./checkpoints/model.hdf5")

"""#continue here from A or B############################################################################ CONTINUE COMPILE AND RUN 
													
# COMPILE the model using mean absolute percentage error as our loss,					
# implying that we seek to minimize the absolute percentage difference					
# between our category *predictions* and the *actual category*						#
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)								
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])			
#model.summary()											#



#checkpoint
filepath="./checkpoints/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')	#
callbacks_list = [checkpoint]



# TRAIN the model
print("[INFO] training model...")									#
model.fit(
	[trainAttrX, trainImagesX], trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=EPOCHS, batch_size=BS,callbacks=callbacks_list)						#


#save model
model.save('./checkpoints/model.hdf5') 									#

#"""#######################################################################################################  can stop here if just want model.hdf5

# make predictions on the testing data
print("[INFO] predicting waste category...")
preds = model.predict([testAttrX, testImagesX])

print (preds.shape)
print preds
print (testAttrX.shape)
print testAttrX
print (testImagesX.shape)
print testImagesX

#"""#   #EVALUATION not compulsory
# compute the difference between the *predicted* waste category and the
# *actual* waste category, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)


# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. waste category: {}, std waste category: {}".format(
	locale.currency(df["category"].mean(), grouping=True),
	locale.currency(df["category"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
#"""#

"""  #CONVERT TO PB functions to freeze and create pb file
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

   
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    graph = session.graph 
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


from keras import backend as K


# Create, compile and train model...

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "/home/kc/Downloads", "kerasMLP.pb", as_text=False)
"""
#print(model.input.op.name)
#print(model.output.op.name)
