import numpy as np
import cv2
import pickle
import os
import glob
import time
import PIL
from PIL import Image
from io import StringIO
from random import randint
from PIL import Image, ImageDraw, ImageFont

import io
from io import StringIO

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.operations import FaceOperations


KEY = os.environ['FACE_SUBSCRIPTION_KEY']
ENDPOINT = os.environ['FACE_ENDPOINT']
PERSON_GROUP_ID = "my-unique-person-group"
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dict_name = {}
with open('dict_name.pickle','rb') as fp:
	dict_name = pickle.load(fp)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def detectFace(image):
	faces = face_client.face.detect_with_stream(image)
	face_ids = []
	final_name = ""
	for face in faces:
		face_ids.append(face.face_id)
		results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
		print('Identifying faces in {}'.format(os.path.basename(image.name)))
		try:
			for person in results:
				getpp = face_client.person_group_person.get(PERSON_GROUP_ID, person.candidates[0].person_id)
				final_name=getpp.name
				'''
				for name in dict_name:
					results = face_client.face.verify_face_to_person(person.face_id,dict_name[name],PERSON_GROUP_ID)
					#print(results)
					if (results.is_identical):
						final_name=name
				'''
		except Exception as e:
			print(e)
		if not results:
			print('No person identified in the person group for faces from {}.'
			.format(os.path.basename(image.name)))
	return final_name

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert the cascade to gray
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces:
		img_item = "my_image"+str(randint(100,200))+".png" 
		roi_gray = gray[y:y+h, x:x+w]
		cv2.imwrite(img_item,roi_gray)
		test_image_array = glob.glob(os.path.join(IMAGES_FOLDER, img_item))
		image = open(test_image_array[0], 'r+b')
		#print(type(image))
		final_name=detectFace(image) 
		with io.BytesIO() as f:
			PIL.Image.fromarray(roi_gray).save(f,"png")
			data = f.getvalue()
		print(data[0])
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
		cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		pilimg = Image.fromarray(cv2img)
		draw = ImageDraw.Draw(pilimg)
		font = ImageFont.truetype('msjhbd.ttc', 20, encoding="utf-8")
		draw.text((x,y-20), final_name,(0, 255, 255), font=font)
		frame = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
		cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
