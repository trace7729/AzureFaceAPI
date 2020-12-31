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

#from face_try import person_idList, person_nameList

KEY = os.environ['FACE_SUBSCRIPTION_KEY']
ENDPOINT = os.environ['FACE_ENDPOINT']
PERSON_GROUP_ID = "my-unique-person-group"
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#img_item = "my_image.jpg"

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

#face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("trainner.yml")
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)))

names_list1 = []
labels = {}
img_item=""
#images_list = []
#images_flist =[]
person_nameList = []
person_idList = []
#with open("labels.pickle", 'rb') as f: # reading bytes, openning labels
#	og_labels = pickle.load(f)
#	labels = {v:k for k,v in og_labels.items()}
for file in os.listdir(BASE_DIR):
	if file.endswith('.png'):
		os.remove(file) 
		print('Remove Image{}'.format(file))
	else:
		pass
with open('person_idList.pickle','rb') as fp:
	person_idList = pickle.load(fp)
with open('person_nameList.pickle','rb') as fp:
	person_nameList = pickle.load(fp)

def detectFace(image):
	faces = face_client.face.detect_with_stream(image)
	face_ids = []
	final_name = ""
	for face in faces:
		face_ids.append(face.face_id)
		results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
		print('Identifying faces in {}'.format(os.path.basename(image.name)))
		if not results:
			print('No person identified in the person group for faces from {}.'
			.format(os.path.basename(image.name)))
		for person in results:
			print('Person for face ID {} is identified in {} with a confidence of {}.'
			.format(person.face_id, os.path.basename(image.name),person.candidates[0].confidence))
			for index, id in enumerate(person_idList):	
				if len(id)==0:
					pass
				else:
					compare = face_client.face.verify_face_to_person(person.face_id, id, PERSON_GROUP_ID)
					if compare.is_identical:
						if compare.confidence >= 0.45 and compare.confidence <= 85:
							getpp = face_client.person_group_person.get(PERSON_GROUP_ID, person_idList[index])
							final_name = getpp.name
	return final_name

labels = dict(zip(person_idList,person_nameList))
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert the cascade to gray
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces:
		#print(x, y, w, h)
		roi_gray = gray[y:y+h, x:x+w]#(ycord_start, ycord_end)
		roi_color = frame[y:y+h, x:x+w]
		#print(type(roi_gray))
		# recongnize
		id_, conf = recognizer.predict(roi_gray)
		if conf >= 45 and conf <= 85:
			#print(id_)
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
			
		#img_item = "my-image1.jpg"
		#cv2.imwrite(img_item, roi_gray) # write image
		color = (255, 0, 0) #BGR 0-255
		stroke = 2
		end_cord_x = x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
'''
cap = cv2.VideoCapture(0)

last_recorded_time = time.time()
while(True):
	curr_time = time.time()
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert the cascade to gray
	detects = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)	
	if len(detects)>=1:
		i = 0
		for d in detects:
			detect = detects[i]
			(x, y, w, h) = detect
			img_item = "my_image"+str(i)+".png" 
			roi_gray = gray[y:y+h, x:x+w]
			cv2.imwrite(img_item,roi_gray)
			test_image_array = glob.glob(os.path.join(IMAGES_FOLDER, img_item))
			image = open(test_image_array[0], 'r+b')
			#name = detectFace(image)
			names_list1.append(name)
			try:
				if (names_list1[-2] == names_list1[-1]):
					names_list1.pop()
			except LookupError:
				pass
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
			#cv2.putText(frame, name, (x+3,y+3), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
			cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			pilimg = Image.fromarray(cv2img)
			draw = ImageDraw.Draw(pilimg)
			font = ImageFont.truetype('msjhbd.ttc', 20, encoding="utf-8")
			draw.text((x,y-20), names_list1[-1],(0, 255, 255), font=font)
			frame = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
			i = i+1
	else:
		pass
		#else:
		#	for (x, y, w, h) in detects:
		#	img_item = "my_image"+str(randint(100,200))+".png" 
		#	images_flist.append(img_item)
		#	roi_gray = gray[y:y+h, x:x+w]
		#	cv2.imwrite(img_item,roi_gray)
		#	test_image_array = glob.glob(os.path.join(IMAGES_FOLDER, img_item))
		#	image = open(test_image_array[0], 'r+b')
		#	name = detectFace(image)
		#	names_list2.append(name)
	cv2.imshow('frame', frame)
	
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()

'''