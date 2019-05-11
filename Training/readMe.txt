1) Main file is Mixed_training.py it loads preprocessed data from dataset.py and then run training on model specified with models.py
2) checkpoints folder MUST be created in advance to store the checkpoints after training, or you waste all training time.
3) Waste dataset has all the waste images in respective category folder and attributes in csv file.


Note: previously training is successful with 4 column attributes, now theres 8 column in csv file needed to be trained. Also the creation of top image paths which will be used for training requires the edit of dataset.py file under load_waste_images_labels() function. This is because we no longer need to extract images and assign them label using their second last path name as the label category.
