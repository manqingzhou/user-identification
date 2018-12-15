
#first try linear svm first, is is kind of easy
import numpy as np
import pandas as pd
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
dataset = pd.read_csv("/Users/zhoumanqing/documents/pycharm/identification/featurelabel.csv",sep=',')
#print (dataset.head(10))

X = dataset.drop('label',axis=1)
y = dataset ['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)   #training = 0.7/testing =0.3
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train,y_train)                 # call the algorithm to train the data

#make prediction
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
