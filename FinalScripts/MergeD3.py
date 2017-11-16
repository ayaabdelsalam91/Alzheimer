import sklearn.metrics as sm
import numpy as np
from pandas import read_csv ,DataFrame
from sklearn.model_selection import train_test_split
from sklearn import svm
import csv

print ("In file MergeD3")

data = '../Finaldata/TADPOLE_D1_D2.csv'
dataframeOR = read_csv(data, names=None)
info=dataframeOR.values
cols = dataframeOR.columns.values
print (info.shape)

data = '../Finaldata/TADPOLE_D3.csv'
dataframeD3 = read_csv(data, names=None)
infoD3=dataframeD3.values
print (infoD3.shape)
cols3 = dataframeD3.columns.values


newData = np.ones((info.shape[0]+infoD3.shape[0], info.shape[1]),dtype=info.dtype)*-4
newData[:info.shape[0],:] = info
print (newData[info.shape[0]:,:].shape)
ind=0

for i in range(info.shape[1]):
	for j in  range(infoD3.shape[1]):
		if(cols[i]==cols3[j]):
			newData[info.shape[0]:,i] = infoD3[:,j]
		elif(cols[i]=="D1" or cols[i]=="D2"):
			newData[info.shape[0]:,i] = 3


df= DataFrame(data= newData , columns = cols)
df.to_csv('../Finaldata/TADPOLE_D1_D2_D3.csv',index =False)