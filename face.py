import numpy as numpy
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
	# for capturing frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting the original frame from into 
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5) 
	''' 
	these values are stated as per documentation,
	and can be altered to see what is happening with them

	'''
	for (x,y,w,h) in faces:
		print (x,y,w,h)
		roi_gray =gray[y:y+h,x:x+w]
		'''
		above contains region of interest i.e. our face, in gray format
		'''
		roi_color = frame[y:y+h,x:x+w]
		img_item_gray = "my_face_gray.png"
		img_item_color = "my_face_color.png"
		cv2.imwrite(img_item_gray, roi_gray)
		cv2.imwrite(img_item_color,roi_color)

		color = (255, 0, 0) #BGR 0-255
		stroke = 2 # how thick we want line of rec.
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y), color, stroke)
	# to display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# after everything being done, release the capture
cap.release()
cv2.destroyAllWindows()