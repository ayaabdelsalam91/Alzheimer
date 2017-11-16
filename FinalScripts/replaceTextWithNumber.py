import sklearn.metrics as sm
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import svm
import csv


def replaceTextWithNumber(InFile,OutFile):
	print ("In file replaceTextWithNumber")


	# data = '../Finaldata/TADPOLE_D1_D2_beforeColDel.csv'
	data = '../Finaldata/'+InFile+'.csv'


	dataframe = read_csv(data, names=None)
	info=dataframe.values
	# print info.shape
	textcolumns = '../Finaldata/translate_text_to_numbers.csv'
	textcolumns = read_csv(textcolumns, names=None)
	textcolumns=textcolumns.values



	for i in range(textcolumns.shape[0]):

		dataframe.replace(textcolumns[i][0], textcolumns[i][1], inplace = True)

	# print dataframe.values.shape
	# dataframe.to_csv('../Finaldata/TADPOLE_processed.csv',index =False)
	dataframe.to_csv('../Finaldata/'+OutFile+'.csv',index =False)
	#dataframe.to_csv('../Finaldata/ToForecastTrain.csv',index =False)
