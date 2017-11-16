import sklearn.metrics as sm
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import svm
import csv

print ("In file deleteUneededcol_BeforeSplit")

data = '../Finaldata/TADPOLE_D1_D2_D3.csv'
dataframe = read_csv(data, names=None)
info=dataframe.values

columns = '../Finaldata/unneeded_column.csv'
columns = read_csv(columns, names=None)
columns=columns.values



processedinfo = dataframe

#Delete columns I know are trash
for i in range(columns.shape[0]):
	processedinfo.drop(columns[i][0] , axis=1,inplace=True)


#Change empty spaces to nan
processedinfo.replace(r'^\s*$', np.nan, regex=True, inplace = True)
#Change empty spaces to nan
processedinfo.replace('-4', np.nan, regex=True, inplace = True)

processedinfo.to_csv('../Finaldata/TADPOLE_D1_D2_beforeColDel.csv',index =False)

