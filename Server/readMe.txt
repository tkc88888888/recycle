1)New waste images(image1top.jpg and image1front.jpg) are saved in Collected_Top_Images and Collected_Front_Images folder respectively.Please modify under 'classify_image' function in 'WasteBinServer.py' to add datetime(same as database) info at the end of every new images coming in to be unique identifier.

2)Occasionally images without waste(latest-without-waste-front.jpg and  latest-without-waste-top.jpg) will be saved. This is fine to be replaced.

3)The kerasmodel.txt and model.hdf5 are the label and model for running prediction.

4)Database should be able to download something like 'new_imagepath_and_attributes_come_in_here_row_by_row.csv' 
