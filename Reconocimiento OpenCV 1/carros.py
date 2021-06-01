import cv2

face_cascade = cv2.CascadeClassifier('cars.xml')

# To capture video from webcam. 
#cap = cv2.VideoCapture(0)
# To use a video file as input 
cap = cv2.VideoCapture('carros4.mp4')

#cap = cv2.imread('carroparqueado.jpg')

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5) #el parametro 1.1 y 4 pueden cambiar
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('img',img)
    k = cv2.waitKey(30)
    if k == 27:  # 27 es eñ ascii para esc
        break
cap.release()