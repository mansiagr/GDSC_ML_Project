import cv2
from cv2 import imread
from cv2 import CascadeClassifier
from cv2 import rectangle

image_path = 'path to filter image'
witch = cv2.imread(image_path)
cv2.imshow('witch', witch)
#get shape of witch
original_witch_h,original_witch_w,witch_channels = witch.shape
witch_gray = cv2.cvtColor(witch, cv2.COLOR_BGR2GRAY)

# create mask and inverse mask of witch
#Note: I used THRESH_BINARY_INV because my image was already on 
#transparent background, try cv2.THRESH_BINARY if you are using a white background
ret, original_mask = cv2.threshold(witch_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)
# example of face detection with opencv cascade classifier



face_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

for image_name in os.listdir(folder):
  image_path = os.path.join(folder, image_name)
  img = cv2.imread(image_path)
  img_h,img_w,img_channels = img.shape
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

  for box in faces:
  # extract
    x, y, w, h = box
    x2, y2 = x + w, y + h
    # draw a rectangle over the pixels
    rectangle(img_gray, (x, y), (x2, y2), (0,0,255), 2)


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
    witch = cv2.resize(witch, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(original_mask, (witch_width,witch_height), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(original_mask_inv, (witch_width,witch_height), interpolation = cv2.INTER_AREA)

    #take ROI for witch from background that is equal to size of witch image
    roi = img[witch_y1:witch_y2, witch_x1:witch_x2]

    #original image in background (bg) where witch is not present
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
    roi_fg = cv2.bitwise_and(witch,witch,mask=mask_inv)
    dst = cv2.add(roi_bg,roi_fg)

    #put back in original image
    img[witch_y1:witch_y2, witch_x1:witch_x2] = dst


    cv2.imshow('image', img) #display image
cv2.waitKey(0) #wait until key is pressed to proceed
cv2.destroyAllWindows() #close all windows



