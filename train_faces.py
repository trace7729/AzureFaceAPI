import os
import numpy as np
from PIL import Image
import cv2
import glob
import pickle
import glob
import time

import re
from collections import defaultdict

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType
current_id = 0
label_ids={}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

#face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#recognizer = cv2.face.LBPHFaceRecognizer_create()

#x_train = []
#y_labels = []

first_label = []
images_name = []
person_idList = []
person_nameList = []
person_name_id = defaultdict(list)
#dict_name = {}

KEY = os.environ['FACE_SUBSCRIPTION_KEY']
ENDPOINT = os.environ['FACE_ENDPOINT']
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

PERSON_GROUP_ID = "my-unique-person-group"
face_client.person_group.delete(person_group_id=PERSON_GROUP_ID)
print("Deleted the person group {} from the source location.".format(PERSON_GROUP_ID))

face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)
print('Person group:', PERSON_GROUP_ID)

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			w = open(path, 'r+b')
			#print(w)
			person_name = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
			person_nameList.append(person_name)
			#c = face_client.person_group_person.create(PERSON_GROUP_ID, person_name)
			#print(person_name)
			#face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, c.person_id, w)
			#print(g)
			#person_id = c.person_id
			#person_idList.append(person_id)
#for person_id in person_idList:
#	print(face_client.person_group_person.get(PERSON_GROUP_ID,person_id))
person_set = set(person_nameList)
#print(person_set)
#print(person_nameList)
#person_list = list(person_set)
for person in person_set:
	group = face_client.person_group_person.create(PERSON_GROUP_ID, person,user_data = person)
	print(group)
	person_id = group.person_id
	person_idList.append(person_id)
dict_name = dict(zip(list(person_set), person_idList))
for name in person_set:
	print(name,dict_name[name])
	for root, dirs, files in os.walk(image_dir):
		if root.endswith(name):
			for file in files:
				if file.endswith("png") or file.endswith("jpg"):
					#print(root)
					path = os.path.join(root,file)
					w = open(path,'r+b')
					face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, dict_name[name],w,user_data = name)
					print(dict_name[name])
'''
for person in person_set:
	c = face_client.person_group_person.create(PERSON_GROUP_ID, person)
	#print(c)
	person_id = c.person_id
	person_idList.append(person_id)
for name, id in zip(person_nameList, person_idList):
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if root.endswith(name):
				print(root)
				if file.endswith("png") or file.endswith("jpg"):
					path = os.path.join(root, file)
					w = open(path, 'r+b')
					g = face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, id, w)
					print(g)
'''
'''
for name, id in zip(person_nameList, person_idList):
	person_name_id[name].append(id)
print(person_name_id)
print('Training the person group...')
'''
# Train the person group
face_client.person_group.train(PERSON_GROUP_ID)
while (True):
    training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
    print("Training status: {}.".format(training_status.status))
    #print()
    if (training_status.status is TrainingStatusType.succeeded):
        break
    elif (training_status.status is TrainingStatusType.failed):
        sys.exit('Training the person group has failed.')
    time.sleep(5)

print(dict_name)
for name in dict_name:
	getpp = face_client.person_group_person.get(PERSON_GROUP_ID, dict_name[name])
	getp = face_client.person_group.get(PERSON_GROUP_ID)
	print(getpp)
	print(getp)
with open('dict_name.pickle','wb') as fp:
	pickle.dump(dict_name, fp)