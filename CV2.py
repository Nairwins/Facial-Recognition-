import cv2
imcap = cv2.VideoCapture(0)
imcap.set(3, 640) # set width as 640
imcap.set(4, 480) # set height as 480
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
while True:
  success, img = imcap.read()
  imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # 1.3 = facteur d'échelle, 5 = voisin minimum pouvant être détecté
  faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
  # réduction de 30% de l’échelle de l’image
  # dessiner un cadre englobant autour du visage
  for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyWindow('face_detect')

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
# Pour chaque personne, entrez un identifiant de visage numérique
face_id = input('\n entrer le ID d\'utilisateur ==> ')
count = 0
while True:
  _, img = cam.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_detector.detectMultiScale(gray, 1.3, 5)
  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, f"clique sur 'n' pour enregistre", (x, y - 10),
  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  cv2.imshow('image', img)
  key = cv2.waitKey(10) & 0xFF
  if key == ord('n'):
    for (x, y, w, h) in faces:
      count += 1
      cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
      print(f"[INFO] Image {count} enregistrée.")
  elif key == ord('q'):
    break
cam.release()
