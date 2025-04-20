import numpy as np
import cv2 #opencv
import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline

import os
import csvLibrary as cl

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pandas as pd

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
import seaborn as sn

import json
import joblib 

#storing paths to all csv files

csv_name_list=[]

for entry in os.scandir("./trainingData"):
    csv_name_list.append(entry.path)
# print(csv_name_list)

#giving nos. to each letter/no.

class_dict = {}
count = 0
# for i in range(10):
#     class_dict[str(i)]=i
    
for i in range(65,91):
    class_dict[chr(i)] = i
# class_dict

# print("Running Data Cleaner")
# import cleaner
# print("\n\nData Cleaned")
print("Running Model Generator")


X=[]
y=[]

for file in csv_name_list:        
    path = file
    # print(path[15])
    csvData = cl.dread(path)

    for rowEntry in csvData:
        # print(rowEntry)
        tempRow = []
        for handPoint, coords in rowEntry.items():
            # print(coords)
            if coords != '':
                coords = list(coords[1:-1].split(","))
                # print(coords)
                # print(float(handPoint), int(coords[0]),int(coords[1]))
                tempRow.extend([float(handPoint), int(coords[0]),int(coords[1])])
            else:
                tempRow.extend([float(handPoint),-600,-600])
        X.append(tempRow)
        y.append(path[15])
# print(X)
# print(y,len(y))

#model generation time 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)

print("Model Generated")

pipe = make_pipeline(StandardScaler(), svm.SVC(gamma='auto',probability=True))
pipe.fit(X_train, y_train)
print("SCORE:",pipe.score(X_test, y_test))


#printing confusion matrix and accuracy table
def getConfusion():
    cm = confusion_matrix(y_test, pipe.predict(X_test))

    axii = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True, xticklabels = axii, yticklabels = axii)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

    # print(classification_report(y_test, pipe.predict(X_test),zero_division=np.nan))

# Save the classes in json and model as a pickle, tflite in a file 
def saveAsPkl(model):
    joblib.dump(model, './artifacts/saved_model.pkl') 
    print("Saved model as saved_model.pkl")
def saveJSON(classes):
    with open("./artifacts/class_dictionary.json","w") as f:
        f.write(json.dumps(classes))
    print("Saved classes as class_dictionary.json")

saveJSON(class_dict)
saveAsPkl(pipe)
getConfusion()