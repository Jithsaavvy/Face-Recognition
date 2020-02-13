#Program for basic face recognition using opencv

#Face Detection
import cv2
import matplotlib.pyplot as plt
import cvlib as cv

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  
  
# capture frames from a camera 
cap = cv2.VideoCapture(0) 
while 1:  
   
    ret, img = cap.read()  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detecting faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x,y,w,h) in faces:

        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray)   
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 

     cv2.imshow('img',img) 
 
      k = cv2.waitKey(30) & 0xff
      if k == 27: 
          break
  
cv2.destroyAllWindows()  

#Face Recognition
image_path = 'sample.png'
im = cv2.imread(image_path)
faces, confidences = cv.detect_face(im)
# Looping through detected faces and add bounding box
for face in faces:
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]    # draw rectangle over face
    cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)# display output

plt.imshow(im)
plt.show()

//Reference: https://docs.opencv.org