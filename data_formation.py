# this utility code is used to generate data in form of images ( from webcam ),
# which to be used to train the LBPH Face Recognizer ( one of the 3 recognizer from openCV)

# Directory tree includes:
# timeLapse { Main Directory }
# |
# |-> images -> { folder, will be generated automatically }
# |            |
# |            |-> apoorve { the dataset having label "apoorve",to be created by code}
# |
# |-> cascades { folder, needed to be copied from official OpenCV package directory using:
# |               (in python terminal, enter -> import cv2, print(cv2.__file__),
# |              this will give address of the data of cascades)  }
# |
# |-> base.py { code }
# |
# |-> data_formation.py { code }
# |
# |-> face.py { code }
# |
# |-> face_train.py { code }
# |
# |-> face_ident.py { code }
# |
# |-> labels.pickle { all the data dumped, being generated }
# |
# |-> trainer.yml { typical weights for using in identification of person }
# |
# |-> README.md

import time
import datetime
import glob
import cv2
import os
import glob

# taking input from user about different labels for which dataset to be made
print(" enter all the labels in next line:")
list_1 = list(map(str, input().split()))

# utility  for, if directory which being stated, doesn't exist, then it creates it
def creator_dir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
creator_dir("images")

# loop for making dataset for each entry in label list
for item in list_1:
    print("recording for label:", item)

    # system_constants
    dataset_dir = "images/"+item
    cap = cv2.VideoCapture(0)
    seconds_duration = 20
    seconds_between_shots = 0.25

    creator_dir(dataset_dir)

    now = datetime.datetime.now()
    finish_time = now + datetime.timedelta(seconds=seconds_duration)

    i = 0  # used for iteration for name

    # to always have current time is less than finish time
    while datetime.datetime.now() < finish_time:
        # capture frame-by-frame
        print("recording")
        ret, frame = cap.read()
        filename = dataset_dir + "/" + str(i) + ".jpg"
        i += 1
        cv2.imwrite(filename, frame)
        time.sleep(seconds_between_shots)
        if cv2.waitKey(20) & 0xFF == ord('q'):  # to just break recording in between
            break
    print("recording for label:", item, " is completed\n")
    print("wait 15 seconds for next label")
    time.sleep(15)
    
    # when everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
print("dataset formation is completed")
