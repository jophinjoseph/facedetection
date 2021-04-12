import os
os.chdir("D:\\trainings\\computer_vision")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2


seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())




def detect_and_show_face_in_image(image_for_face_detection, face_cascade_classifier, eye_cascade_classifier, sz):
    # Convert to gray
    gray = cv2.cvtColor(image_for_face_detection, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade_classifier.detectMultiScale(gray, scaleFactor = 1.3, # jump in the scaling factor, as in, if we don't find an image in the current scale, the next size to check will be, in our case, 1.3 times bigger than the current size.
                                                     minNeighbors = 5, minSize=(30, 30))
    str_msg_on_image = 'faces: ' + str(len(faces)); eyes_count = 0
    for (x,y,w,h) in faces:
      t
        if h <= 0 or w <= 0: pass

        # get face rectangle
        image_for_face_detection = cv2.rectangle(image_for_face_detection,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

        # detect eye
        eyes = eye_cascade_classifier.detectMultiScale(roi_gray)


        roi_color = image_for_face_detection[y:y+h, x:x+w]
        # get eye rectangle
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyes_count = eyes_count + 1
   e
    str_msg_on_image = str_msg_on_image + ', eyes: ' + str(eyes_count)
    cv2.putText(image_for_face_detection, str_msg_on_image, (sz[0]-200, sz[1]-25), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)

    return(image_for_face_detection)
# end of detect_and_show_face_in_image


face_cascade_classifier = cv2.CascadeClassifier('./model/haarcascades/haarcascade_frontalface_default.xml')
if face_cascade_classifier.empty():
    print('Missing face classifier xml file')

eye_cascade_classifier = cv2.CascadeClassifier('./model/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
if eye_cascade_classifier.empty():
    print('Missing eye classifier xml file')


image_for_face_detection = cv2.imread('./image_data/face_3.png')
sz = (image_for_face_detection.shape[1], image_for_face_detection.shape[0])


image_for_face_detection = detect_and_show_face_in_image(image_for_face_detection, face_cascade_classifier, eye_cascade_classifier, sz)

cv2.imshow('image_for_face_detection',image_for_face_detection)
cv2.waitKey(0); cv2.destroyAllWindows()

