from tkinter.constants import S
import numpy as np
from backpropagation import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import os
import cv2
import pickle

from prepare_dataset import prepare_dataset

class nn_aksara:
  
  @staticmethod
  def normalize_1d(image):
    image_1d = image.flatten()
    f = lambda image_1d: image_1d/255
    image_normalize_1d = f(image_1d) 
    return image_normalize_1d

  @staticmethod
  def training(epoch):
    dataset_dir = "./dataset/"

    labels = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]
    dataset_count = sum([len(files) for r, d, files in os.walk("./dataset/")])

    datasetX = np.zeros((dataset_count,900))
    datasetY = np.zeros((dataset_count,20))

    hanacara_loc = ["ha","na","ca","ra","ka",
                    "da","ta","sa","wa","la",
                    "pa","dha","ja","ya","nya",
                    "ma","ga","ba","tha","nga"]

    print("Dataset count : {}".format(dataset_count))
    x=0
    for label in labels:
        for filename in os.listdir(dataset_dir+label):
            img_to_preprocess = cv2.imread(dataset_dir+label+"/"+filename)
            img_preprocessed = prepare_dataset.preprocess(img_to_preprocess)
            normalize = np.asarray(nn_aksara.normalize_1d(img_preprocessed))
            datasetX[x,] = normalize
            arrayY = np.zeros((20,))
            arrayY[hanacara_loc.index(label)] = 1
            datasetY[x,] = arrayY
            x+=1
            print("Preprocessing data {:.2f}% complete.".format((x/dataset_count)*100),end='\r')

    (trainX, testX, trainY, testY) = train_test_split(datasetX,datasetY, stratify=datasetY, test_size=0.32, random_state=1)

    model = NeuralNetwork([trainX.shape[1],120,60,20])
    #model = MLPClassifier(hidden_layer_sizes=(120,60),random_state=100,n_iter_no_change=epoch,max_iter=epoch,verbose=True)

    print("Training....")
    print("[INFO] {}".format(model))
    #model.fit(trainX,trainY)
    model.fit(trainX,trainY,epoch)
    '''
    Training....g data 100.00% complete.
    [INFO] NeuralNetwork:900-120-60-20
    [INFO] epoch=1, loss=161.5051790
    [INFO] epoch=10, loss=158.7165370
    [INFO] epoch=20, loss=141.4222750
    [INFO] epoch=30, loss=87.9702123
    [INFO] epoch=40, loss=37.4046784
    [INFO] epoch=50, loss=13.5584285
    [INFO] epoch=60, loss=6.6294976
    [INFO] epoch=70, loss=4.0655959
    [INFO] epoch=80, loss=3.0324983
    [INFO] epoch=90, loss=2.2622402
    [INFO] epoch=100, loss=1.7263422
    Evaluating...
                  precision    recall  f1-score   support

               0       0.80      0.50      0.62         8
               1       1.00      1.00      1.00         8
               2       1.00      0.75      0.86         8
               3       1.00      0.88      0.93         8
               4       0.67      0.75      0.71         8
               5       0.89      1.00      0.94         8
               6       0.78      0.88      0.82         8
               7       0.80      1.00      0.89         8
               8       0.86      0.75      0.80         8
               9       1.00      0.75      0.86         8
              10       0.83      0.62      0.71         8
              11       0.88      0.88      0.88         8
              12       0.89      1.00      0.94         8
              13       0.80      1.00      0.89         8
              14       1.00      0.88      0.93         8
              15       0.73      1.00      0.84         8
              16       0.80      1.00      0.89         8
              17       1.00      1.00      1.00         8
              18       1.00      0.62      0.77         8
              19       0.70      0.88      0.78         8

         accuracy                           0.86       160
        macro avg       0.87      0.86      0.85       160
     weighted avg       0.87      0.86      0.85       160
    '''

    pickle.dump(model, open("model-aksara.pickle",'wb'))

    print("Evaluating...")
    predictions = model.predict(testX)
    predictions = predictions.argmax(axis=1)
    print(classification_report(testY.argmax(axis=1),predictions))
    
  @staticmethod
  def prediction(image)->str:
    hanacara_loc = ["ha","na","ca","ra","ka",
                    "da","ta","sa","wa","la",
                    "pa","dha","ja","ya","nya",
                    "ma","ga","ba","tha","nga"]
    model = pickle.load(open("model-aksara.pickle",'rb'))

    img_preprocessed = prepare_dataset.preprocess(image)
    img_1d = nn_aksara.normalize_1d(img_preprocessed)
    predict_arr = np.zeros((1,900))
    predict_arr[0,] = img_1d

    predictions = model.predict(predict_arr)
    prediction_str = hanacara_loc[np.argmax(predictions[0])]
    return prediction_str
