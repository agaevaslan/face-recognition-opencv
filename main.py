#************************
#***итогова€ программа***
#************************
import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognition_trainer/trainer.yml')
cascadePath = "1.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#индикатор id пользовател€
id = 0

# имена пользователей, св€занные по id с 0...N
names = ['Nobody', 'Evgenii', 'Aslan'] 

# инициализируем веб-камеру
cam = cv2.VideoCapture(0)
cam.set(3, 640) # фиксируем ширину кадра
cam.set(4, 480) # фиксируем высоту кадра

# ”станавливаем минимальные размеры распознаваемого лица
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # јнализируем коэффициент несовпадени€. ќн должен быть меньше 100
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # 'ESC' дл€ выхода из окна просмотра видео
    if k == 27:
        break

# ¬ыходим из программы, закрываем окна
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
