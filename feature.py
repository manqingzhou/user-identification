import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller

#make time index,paerse_Dates specifies which column is date_time information
cwd2 = "/Users/zhoumanqing/Documents/pycharm/identification/normal62.csv"
dataFrame = pd.read_csv(cwd2, delimiter=',', parse_dates=['time'], index_col='time')
#create a temporary DataFrame with the index as a column usinng
dataFrame.reset_index()
dataFrame.reset_index().plot(y=['acc-x','acc-y','acc-z'],x='time')
plt.show()


rolmean = dataFrame.rolling(3000).mean()
rolstd  = dataFrame.rolling(3000).std()
rolvar  = dataFrame.rolling(3000).var()
#solve the problem  nan ,use inplace=true,if true, do operation inplace and return none
rolmean.dropna(inplace=True)
rolstd.dropna(inplace=True)
rolvar.dropna(inplace=True)
#print (rolmean.head(1000))
#print (rolstd.head(1000))
#print (rolvar.head(1000))

#with open()

rolmean.to_csv("/Users/zhoumanqing/Documents/pycharm/identification/mean6.csv")
rolstd.to_csv("/Users/zhoumanqing/Documents/pycharm/identification/std6.csv")
rolvar.to_csv("/Users/zhoumanqing/Documents/pycharm/identification/var6.csv")
a = pd.read_csv("/Users/zhoumanqing/Documents/pycharm/identification/mean6.csv")
b = pd.read_csv("/Users/zhoumanqing/Documents/pycharm/identification/std6.csv")
c = pd.read_csv("/Users/zhoumanqing/Documents/pycharm/identification/var6.csv")
#conbine all the features together
result = pd.concat([a,b,c],axis=1)
result.drop(['time'],axis=1,inplace=True)
result.to_csv("/Users/zhoumanqing/Documents/pycharm/identification/user6.csv")
print (result.head(10))


#plot rolling statistics:
#orig = plt.plot(dataFrame,color='blue',label='original')
mean = plt.plot(rolmean,color='red',label='rolling mean')
std  = plt.plot(rolstd,color='black',label='rolling std')
#location string'best'	0
#'upper right'	1
#'upper left'	2
#'lower left'	3
#'lower right'	4
#'right'	5
#'center left'	6
#'center right'	7
#'lower center'	8
#'upper center'	9
#'center'	10
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

#next step is to discard useless NaN,in fact i can do it manually,but i will not
