# this file contains code to use weight from training, to identify correct person in
# real time

import numpy as numpy
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    # for inverting the dictionary
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # for capturing frame-by-frame
    ret, frame = cap.read()
    # converting the original frame from into
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    ''' 
	these values are stated as per documentation,
	and can be altered to see what is happening with them

	'''
    for (x, y, w, h) in faces:
        # print (x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        '''
		above contains region of interest i.e. our face, in gray format
		'''
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])

            # for putting text
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)
        img_item_gray = "my_face_gray.png"
        img_item_color = "my_face_color.png"
        cv2.imwrite(img_item_gray, roi_gray)
        cv2.imwrite(img_item_color, roi_color)

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2  # how thick we want line of rec.
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # to display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        ret, frame = cap.read()
        frame = cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        frame = cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        filename = "sample_detection.jpg"
        cv2.imwrite(filename, frame)
        break

# after everything being done, release the capture
cap.release()
cv2.destroyAllWindows()
