import cv2
import os
from cv2 import imread
from cv2 import CascadeClassifier
from cv2 import rectangle
import numpy as np
import matplotlib.pyplot as plt

image_path = '/content/drive/MyDrive/gdsc/witch (1).png'
witch = cv2.imread(image_path)
cv2.imshow('image', witch)
#get shape of witch
original_witch_h,original_witch_w,witch_channels = witch.shape
witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)

# create mask and inverse mask of witch
ret, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Haar cascades classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video file
video_capture = cv2.VideoCapture('/content/drive/MyDrive/gdsc/Copy of face_demographics.mp4')
# Check if the video capture object is open
if not video_capture.isOpened():
    print("Could not open video file.")
    exit()

# Loop over the frames of the video
while True:
    # Read the next frame from the video
    ret, frame = video_capture.read()

    # Check if the frame was successfully read
    if not ret:
        print("Could not read frame from video file.")
        break

    # Display the resulting frame
    img_h,img_w = frame.shape[:2]
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #witch size in relation to face by scaling
        witch_width = int(1.5 * face_w)
        witch_height = int(witch_width * original_witch_h / original_witch_w)
        
        #setting location of coordinates of witch
        witch_x1 = face_x2 - int(face_w/2) - int(witch_width/2)
        witch_x2 = witch_x1 + witch_width
        witch_y1 = face_y1 - int(face_h*1.25)
        witch_y2 = witch_y1 + witch_height 

        #check to see if out of frame
        if witch_x1 < 0:
            witch_x1 = 0
        if witch_y1 < 0:
            witch_y1 = 0
        if witch_x2 > img_w:
            witch_x2 = img_w
        if witch_y2 > img_h:
            witch_y2 = img_h

        #Account for any out of frame changes
        witch_width = witch_x2 - witch_x1
        witch_height = witch_y2 - witch_y1

        #resize witch to fit on face
        if (witch_width,witch_height) is not tuple([0, 0]):

          print(witch_width,witch_height)
          witch = cv2.resize(witch, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
          mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
          mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
        else:
          # (witch_width,witch_height) = None
          print(witch_width,witch_height)
          witch = cv2.resize(witch, None , interpolation = cv2.INTER_AREA)
          mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
          mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)

        #take ROI for witch from background that is equal to size of witch image
        roi = frame[witch_y1:witch_y2, witch_x1:witch_x2]

        #original image in background (bg) where witch is not present
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(witch,witch,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        frame[witch_y1:witch_y2, witch_x1:witch_x2] = dst


    cv2.imshow('image',frame) #display image
    cv2.waitKey(0) #wait until key is pressed to proceed
    cv2.destroyAllWindows() #close all windows


    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
