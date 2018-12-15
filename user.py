import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import scipy
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import matplotlib.lines as mlines

#tell the path
cwd = "/Users/zhoumanqing/Documents/pycharm/identification/6.csv"
#transfer into csv
dataFrame = pd.read_csv(cwd, delimiter=',')
#add columns
dataFrame.columns = ["time","acc-x","acc-y","acc-z"]
print(dataFrame)
plt.plot(x='time',y='acc-x')
plt.plot(x='time',y='acc-y')
plt.plot(x='time',y='acc-z')
plt.show()
array =dataFrame.values
#choose from 1to3 columns
X = array[:,1:4]
Y = array[:,3]
#normalization
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
#only need 3 numbers behind the point
np.set_printoptions(precision=3)
#print(rescaledX[0:2999,:])
df = pd.DataFrame(rescaledX,columns=['acc-x','acc-y','acc-z'])
export_csv=df.to_csv("/Users/zhoumanqing/Documents/pycharm/identification/normal61.csv", sep=',')
print(df)
















