import pandas as pd
import numpy as np
import glob
import cv2
import os
from imutils import paths
from sklearn.preprocessing import MinMaxScaler

def load_waste_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["category","weight", "length", "width", "height","imagepath"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

	# return the data frame
	#print df	
	return df

def process_waste_attributes(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["weight", "length", "width", "height"]

	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainX = cs.fit_transform(train[continuous])
	testX = cs.transform(test[continuous])

	# return the concatenated training and testing data
	
	#print(trainX.shape)
	return (trainX, testX)

def load_waste_images_labels(df, inputPath):
	# initialize our images array and label name array
	images = []
	labels = []
	subdir = ['aluminium','cardboard','glass','paper','plastic','trash']

	# loop over the indexes of the wastes
	
	for folder in subdir:
		# find the four images for the waste and sort the file paths,
		# ensuring the four are always in the *same order*
		basePath = os.path.sep.join([inputPath,"{}".format(folder)])
		wastePaths = sorted(list(paths.list_images(basePath)))
		
		for wastePath in wastePaths:
			# load the image, resize it to be 224 224, and then
			# update the list of input images
			image = cv2.imread(wastePath)
			image = cv2.resize(image, (224, 224))
			images.append(image)
			
			# extract the class label from the image path and update the
			# labels list
			label = wastePath.split(os.path.sep)[-2]
			if label == 'trash':
				labels.append(0)

			elif label == 'plastic':
				labels.append(1)

			elif label == 'paper':
				labels.append(2)

			elif  label == 'glass':
				labels.append(3)

			elif  label == 'cardboard':
				labels.append(4)

			elif label == 'aluminium':
				labels.append(5)
	
	
	# return our set of images
	return (images,labels)
