# this file includes code to train the classifier on the image dataset
import os
from PIL import Image # PIL : python image lib.
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
'''
whereever the orginal file is saved, the above command
returns the address of it, and is true for any OS system
'''
image_dir = os.path.join(BASE_DIR, "images") # going to take base directory and will return images

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			# label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
			'''
			above command returns same as this one, returning name of folder of image
			which to be used as label
			'''
			label = os.path.basename(root).replace(" ","-").lower()
			# print(label, path)

			# below part, to add images along with their new labels,
			# if they are not, in existing dictionary of labels
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			# print(label_ids)
			# y_labels.append(label) # some number
			# x_train.append(path) # verify this image, turn into a NUMPY array, GRAY
			pil_image = Image.open(path).convert("L") # for coverting into GRAYSCALE
			
			'''
			to resize all the images to a fix size, so that no problem in training might occur
			'''
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS )
			image_array = np.array(final_image,"uint8") # uint8 as type
			# print(image_array)
			'''
			below now, we used to detect the region of interest i.e. faces
			in the gray picture, which is in form or numpy array
			'''
			faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)
			for (x,y,w,h) in faces:
				# print (x,y,w,h)
				roi = image_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

# print(y_labels) # this dictionary include labels and a number as id associated with it.
# print(x_train) # this array includes the parts of numpy arrays of each images, which represent the RoI for that image

with open("labels.pickle","wb") as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")