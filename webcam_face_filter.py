import cv2
import face_recognition
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

video_capture = cv2.VideoCapture(0)

face_locations = []

import cv2
from cv2 import imread

image_path = '/Users/mac/Desktop/ecs189/witch (2).png'
witch = cv2.imread(image_path)
# cv2.imshow('witch',witch)
#get shape of witch
original_witch_h,original_witch_w,witch_channels = witch.shape
witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)

# create mask and inverse mask of witch
#Note: I used THRESH_BINARY_INV because my image was already on 
#transparent background, try cv2.THRESH_BINARY if you are using a white background
ret, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rgb_frame = frame[:, :, ::-1]
    img_h, img_w = frame.shape[:2]
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        face_w = right-left
        face_h = bottom-top
        face_x1 = left
        face_x2 = face_x1 + face_w
        face_y1 = top
        face_y2 = face_y1 + face_h
        
        witch_width = int(1.5 * face_w)
        witch_height = int(witch_width * original_witch_h / original_witch_w)
        witch_x1 = face_x2 - int(face_w/2) - int(witch_width/2)
        witch_x2 = witch_x1 + witch_width
        witch_y1 = face_y1 - int(face_h*1.25)
        witch_y2 = witch_y1 + witch_height 
        if witch_x1 < 0:
            witch_x1 = 0
        if witch_y1 < 0:
            witch_y1 = 0
        if witch_x2 > img_w:
            witch_x2 = img_w
        if witch_y2 > img_h:
            witch_y2 = img_h

        witch_width = witch_x2 - witch_x1
        witch_height = witch_y2 - witch_y1

        #resize witch to fit on face
        witch = cv2.resize(witch, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        # Display the resulting image
        roi = frame[witch_y1:witch_y2, witch_x1:witch_x2]

        #original image in background (bg) where witch is not
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(witch,witch,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        frame[witch_y1:witch_y2, witch_x1:witch_x2] = dst
        cv2.imshow('Video', frame)
        # Hit ‘q’ on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

video_capture.release()
cv2.destroyAllWindows()
