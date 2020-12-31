import os
import numpy as np
from PIL import Image
import cv2
import pickle
current_id = 0
label_ids={}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

x_train = []
y_labels = []
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
			#print(label, path)
			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id # label is person's name
				current_id += 1
			id_ = label_ids[label] # number id for person's name
			#print(label_ids)
			#y_labels.append(label) # keep label
			#x_train.append(path) # keep path, turn into a numpy array, gray
			
			pil_image = Image.open(path).convert("L") # pixel value to grayscale\
			print(pil_image)
			image_array = np.array(pil_image, "uint8") # grayscale to numpy array
			
			#print(image_array)
			#region of interest
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				#creating training labels
				y_labels.append(id_)
#print(y_labels)
#print(x_train)
with open("labels.pickle", 'wb') as f: # writing bytes, saving labels
	pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels)) # training data actual numpy array
# convert y_labels into numpy array
recognizer.save("trainner.yml")