import numpy as np
import cv2
import pickle
import time

from PIL import Image, ImageDraw, ImageFont
name = ""
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f: # reading bytes, openning labels
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
	
cap = cv2.VideoCapture(0)
last_recorded_time = time.time()
while(True):
	curr_time = time.time()
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
			cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			pilimg = Image.fromarray(cv2img)
			draw = ImageDraw.Draw(pilimg)
			font = ImageFont.truetype('msjh.ttc', 20, encoding="utf-8")
			draw.text((x,y), name,(255,0,0), font=font)
			frame = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
			#cv2.imshow("print chinese to image", img_OpenCV)
			#cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		color = (255, 0, 0) #BGR 0-255
		stroke = 2
		end_cord_x = x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
	
		if curr_time - last_recorded_time >= 3.0:
			img_item = "thatthisthat-image1.jpg"
			cv2.imwrite(img_item,roi_gray) # write image
			last_recorded_time = curr_time
	cv2.imshow('frame', frame)	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()