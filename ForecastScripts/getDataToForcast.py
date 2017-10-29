import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from sklearn import svm
import csv


# data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/TADPOLE_D1_D2_beforeColDel.csv'
# data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/ToForecast.csv'
data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/ToForecastTrain.csv'
dataframe = read_csv(data, names=None)
ToForecast=dataframe.values
print ToForecast.shape
data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/Local_TrainingAfterImpHD.csv'

dataframe = read_csv(data, names=None)
info=dataframe.values
print info.shape
index = []
for i in range(info.shape[0]):
	for j in range(ToForecast.shape[0]):
		if(info[i,0]==ToForecast[j,0] and info[i,1]==ToForecast[j,1]):
			index.append(i)


Traininfo = info[index]


data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/Local_TestingAfterImpHD.csv'
dataframe = read_csv(data, names=None)
info=dataframe.values
print info.shape
index = []
for i in range(info.shape[0]):
	for j in range(ToForecast.shape[0]):
		if(info[i,0]==ToForecast[j,0] and info[i,1]==ToForecast[j,1]):
			index.append(i)


Testinginfo = info[index]


info  = np.vstack((Traininfo,Testinginfo))
print info.shape
dataframe = DataFrame(info , columns = dataframe.columns.values)

dataframe.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/ToForecastLeaderBoardTrain.csv',index =False)

