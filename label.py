mport numpy as np
import pandas as pd
import csv


cwd ="/Users/zhoumanqing/documents/pycharm/identification/features.csv"
dataFrame = pd.read_csv(cwd, delimiter=',')
dataFrame.columns = ["acc-x-mean","acc-y-mean","acc-z-mean","acc-x-std","acc-y-std","acc-z-std","acc-x-var","acc-y-var","acc-z-var"]
idx = 0 #first column
dataFrame.insert(loc=idx, column='label',value=0)


dataFrame.label.iloc[0:500]=0
dataFrame.label.iloc[501:1001]=1
dataFrame.label.iloc[1002:1502]=2
dataFrame.label.iloc[1503:2003]=3
dataFrame.label.iloc[2004:2504]=4
dataFrame.label.iloc[2505:2995]=5
print (dataFrame.head(600))

dataFrame.to_csv("/Users/zhoumanqing/documents/pycharm/identification/featurelabel.csv",sep=',')
