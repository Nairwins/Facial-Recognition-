import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
faceCascade = cv2.CascadeClassifier (cv2.data.haarcascades +
"haarcascade_frontalface_default.xml")
# les nomes relative aux ids: exemple ==> Pers1: id=1, etc
names = ['None', 'Pers1', 'Pers2', 'Pers3', 'Pers4', 'Pers5']
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
  while True:
  _, img =cam.read()
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale( gray, scaleFactor = 1.3, minNeighbors = 5)
  for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    # Vérifiez si la confiance est inférieure à 100 ==> "0" correspond parfait match
    if (confidence < 100):
      id = names[id]
    else:
      id = "inconnu"
      confidence = " {0}%".format(round(100 - confidence))
      cv2.putText(img,str(id),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
      cv2.putText(img,str(confidence),(x+5,y+h-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0), 1)
cv2.imshow('camera',img)
k = cv2.waitKey(10) & 0xff # Appuyez sur 'ESC' pour quitter la vidéo
if k == 27:
break
