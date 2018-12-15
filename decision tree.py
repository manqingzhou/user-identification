#second try decision tree, is is kind of easy and i cannot do for loop
import numpy as np
import pandas as pd
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
dataset = pd.read_csv("/Users/zhoumanqing/documents/pycharm/identification/featurelabel.csv",sep=',')
#print (dataset.head(10))

X = dataset.drop('label',axis=1)
y = dataset ['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)   #training = 0.8/testing =0.2
svclassifier = DecisionTreeClassifier()
svclassifier.fit(X_train,y_train)                 # call the algorithm to train the data

#make prediction
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
