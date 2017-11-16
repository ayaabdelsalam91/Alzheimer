import sklearn.metrics as sm
import numpy as np
from pandas import read_csv , DataFrame
from sklearn.model_selection import train_test_split
from sklearn import svm
import csv

def split(infile ,trainingOut,testingOut,D3Out):
	print ("In file splitFiles")

	data = '../Finaldata/'+infile+'.csv'
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	print(info.shape)

	training = dataframe[dataframe['D1']==1]

	testing = dataframe[dataframe['D2']==1]
	D3=dataframe[dataframe['D2']==3]


	train= DataFrame(data= training.values , columns = dataframe.columns.values)
	train.drop(['D1','D2'] , axis=1,inplace=True)
	print(train.shape)

	test= DataFrame(data= testing.values , columns = dataframe.columns.values)
	test.drop(['D1','D2'] , axis=1,inplace=True)
	print(test.shape)

	D3= DataFrame(data= D3.values , columns = dataframe.columns.values)
	D3.drop(['D1','D2'] , axis=1,inplace=True)
	print(D3.shape)

	D3.to_csv('../Finaldata/'+D3Out+'.csv', index =False)
	test.to_csv('../Finaldata/'+testingOut+'.csv', index =False)
	train.to_csv('../Finaldata/'+trainingOut+'.csv', index =False)
