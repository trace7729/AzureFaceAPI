import numpy as np
import cv2
import pickle
import time
import requests
import os
import io
import PIL
from PIL import Image, ImageDraw, ImageFont
from io import StringIO

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.operations import FaceOperations

face_ids = []
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
faceId=''

KEY = os.environ['FACE_SUBSCRIPTION_KEY']
ENDPOINT = os.environ['FACE_ENDPOINT']
PERSON_GROUP_ID = "my-unique-person-group"
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dict_name = {}
with open('dict_name.pickle','rb') as fp:
	dict_name = pickle.load(fp)

API_KEY = os.environ['FACE_SUBSCRIPTION_KEY'] 
endpoint = 'https://westus.api.cognitive.microsoft.com/face/v1.0/detect'
args = {'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,emotion'}
headers = {'Content-Type': 'application/octet-stream',
           'Ocp-Apim-Subscription-Key': API_KEY}

cap = cv2.VideoCapture(0)
last_recorded_time = time.time()
while(True):
	curr_time = time.time()
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert the cascade to gray
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces:
		roi_gray = gray[y:y+h, x:x+w]#(ycord_start, ycord_end)
		with io.BytesIO() as f:
			PIL.Image.fromarray(roi_gray).save(f,"png")
			data = f.getvalue()
		try:		
			response = requests.post(data=data,url=endpoint,headers=headers,params=args)
			attr_label=''
			for face in response.json():
				faceId = face['faceId']
				fattr = face['faceAttributes']
				disp = {'gender': fattr['gender'], 'age':fattr['age']}
				emotion_dict = fattr['emotion']
				emotion = max(emotion_dict.keys(), key = lambda k: emotion_dict[k]) 
				disp['emotion'] = emotion
				#disp['name']="unknown"
				#face_ids.append(faceId)
				#if curr_time - last_recorded_time >=3:
				#getpp1=""
				'''
				try:
					#results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
					for name in dict_name:
						results = face_client.face.verify_face_to_person(faceId,dict_name[name],PERSON_GROUP_ID)
						if (results.is_identical):
							disp['name']=name
						break
					#for person in results:
					#	getpp = face_client.person_group_person.get(PERSON_GROUP_ID,person.candidates[0].person_id)
					#	getpp1=getpp.name
				except Exception as e:
					print(e)
					pass
				'''
				#disp['name']=getpp1
				#last_recorded_time = curr_time
				#face_ids.append(faceId)
				#results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
				for i, k in enumerate(disp):
					attr_label += "{0}: {1} \n".format(k, disp[k])
			for i, line in enumerate(attr_label.split('\n')):
				y1 = y-i*25
				color_atr = (255, 0, 0)
				#frame = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
				stroke_atr = 2
				font_atr = cv2.FONT_HERSHEY_SIMPLEX
				
				cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				pilimg = Image.fromarray(cv2img)
				draw = ImageDraw.Draw(pilimg)
				font = ImageFont.truetype('msjh.ttc', 20, encoding="utf-8")
				draw.text((x,y1), line,(255,0,0), font=font)
				frame = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
				
				#cv2.putText(frame, line, (x,y1), font_atr, 1, color_atr, stroke_atr, cv2.LINE_AA) 
				
				color_frame = (255, 255, 255) #BGR 0-255
				stroke_frame = 2
				# draw rectangle
				end_cord_x = x+w
				end_cord_y = y+h
				cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color_frame, stroke_frame)
				#attr_label =''
		except (TypeError):
			pass
		'''
		#if curr_time - last_recorded_time >= 3:
			#img_item = "thatthisthat-image1.jpg"
			#cv2.imwrite(img_item,roi_gray) # write image
			#face_ids.append(faceId)
			#results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
			#for person in results:
				#print('Person for face ID {} is identified with a confidence of {}.'
				#.format(person.face_id,person.candidates[0].confidence))
				#print(person.candidates[0].person_id)
				#print(dict_name[person.candidates[0].person_id])
				#verify = face_client.face.verify_face_to_person(person.face_id,person.candidates[0].person_id,PERSON_GROUP_ID)
				#getpp = face_client.person_group_person.get(PERSON_GROUP_ID,person.candidates[0].person_id)
				#disp['name']=getpp.name
			#for name in dict_name:
				#verify = face_client.face.verify_face_to_person(person.face_id,dict_name[name],PERSON_GROUP_ID)
				#if verify.confidence>= 0.45 and verify.confidence <= 85:
					#print(verify,name)
					#break
				#verify = face_client.face.verify(person.face_id, 
				#compare = face_client.face.identify(person.face_id, id, PERSON_GROUP_ID)
				#last_recorded_time = curr_time
				#else:
				#pass
		'''
	#cv2.namedWindow("frame",cv2.WND_PROP_FULLSCREEN)
	#cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
for file in os.listdir(BASE_DIR):
	if file.endswith('.png'):
		os.remove(file) 
		print('Remove Image{}'.format(file))