import numpy as np
import cv2
import pickle
import time
import PIL
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f: # reading bytes, openning labels
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
'''
person_nameList = []
person_idList = []
with open('person_idList.pickle','rb') as fp:
	person_idList = pickle.load(fp)
with open('person_nameList.pickle','rb') as fp:
	person_nameList = pickle.load(fp)
labels = dict(zip(person_idList,person_nameList))
'''	
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
		#print(id_)
		if conf >= 60 and conf <= 85:
			#print(id_)
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			#cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
			cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			pilimg = Image.fromarray(cv2img)
			draw = ImageDraw.Draw(pilimg)
			font = ImageFont.truetype('msjhbd.ttc', 20, encoding="utf-8")
			draw.text((x,y-30), name,(0, 255, 255), font=font)
			frame = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
			
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