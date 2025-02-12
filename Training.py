import cv2
import numpy as np
from PIL import Image
import os
# Path de dataset
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
def getImagesAndID(path):
  imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
  faceSamples=[]
  ids = []
  for imagePath in imagePaths:
    PIL_img = Image.open(imagePath)
    img_numpy = np.array(PIL_img,'uint8')
    id = int(os.path.split(imagePath)[-1].split(".")[1])
    faceSamples.append(img_numpy)
    ids.append(id)
  return faceSamples,ids
  
faces,ids = getImagesAndID(path)
recognizer.train(faces, np.array(ids))
# Enregistrez le modèle dans trainer.yml
recognizer.write('trainer.yml')
# Imprimer le nombre de visages formés et terminer le programme
print("\n {0} visages entraînés. Programme terminier".format(len(np.unique(ids))))
