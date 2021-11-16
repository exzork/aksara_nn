import numpy
from backpropagation import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import random
import cv2

def normalize_1d(image):
  image_1d = image.flatten()
  f = lambda image_1d: image_1d/255
  image_normalize_1d = f(image_1d) 
  return image_normalize_1d

dataset_dir = "./dataset_30x30/"

labels = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]

datasetX = numpy.zeros((100,900))
datasetY = numpy.zeros((100,20))

hanacara_loc = ["ha","na","ca","ra","ka",
                "da","ta","sa","wa","la",
                "pa","dha","ja","ya","nya",
                "ma","ga","ba","tha","nga"]

x=0
for label in labels:
    for filename in os.listdir(dataset_dir+label):
        img_gray = cv2.imread(dataset_dir+label+"/"+filename,0)
        normalize = numpy.asarray(normalize_1d(img_gray))
        datasetX[x,] = normalize
        arrayY = numpy.zeros((20,))
        arrayY[hanacara_loc.index(label)] = 1
        datasetY[x,] = arrayY
        x+=1

(trainX, testX, trainY, testY) = train_test_split(datasetX,datasetY, test_size=0.2)

nn = NeuralNetwork([trainX.shape[1],120,60,20])
print("Training....")
print("[INFO] {}".format(nn))
nn.fit(trainX,trainY,epoch=1000)

print("Evaluating...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(predictions)
print(classification_report(testY.argmax(axis=1),predictions))

