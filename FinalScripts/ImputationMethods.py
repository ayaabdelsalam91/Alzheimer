import sklearn.metrics as sm
import numpy as np
from pandas import read_csv
import csv
import math
# from fancyimpute.knn import KNN

from sklearn.linear_model import LinearRegression
# import skmice
def fillwithMean(df):
	temp_df = df
	df.to_csv('../Finaldata/tempdf_with_Missing_Data.csv')
	print ( sum(df.isnull().sum(axis=0).tolist()))
	for i in range (0,temp_df.shape[1]):
		if("Unnamed" not in  temp_df.columns[i] and "RID" not in  temp_df.columns[i]):
				print ( i )
			#print df[df.columns[i]].unique()
				if(temp_df[temp_df.columns[i]].unique().shape[0] < 22):
					# print "Cat" , temp_df[temp_df.columns[i]].mode()
					if(not temp_df[temp_df.columns[i]].mode().empty):
						mode = temp_df[temp_df.columns[i]].value_counts().idxmax()
						temp_df[temp_df.columns[i]] = temp_df.groupby('RID')[temp_df.columns[i]].apply(lambda x: x.fillna(x.value_counts().idxmax() if x.value_counts().max() >=1 else mode , inplace = False))
				else:
					temp_df[temp_df.columns[i]] = temp_df.groupby('RID')[temp_df.columns[i]].transform(lambda x: x.fillna(x.mean()))
					temp_df[temp_df.columns[i]] = temp_df[temp_df.columns[i]].fillna(temp_df[temp_df.columns[i]].mean())
	temp_df = temp_df.fillna(value = -10)
	print ( sum(temp_df.isnull().sum(axis=0).tolist()))
	return temp_df

def HotDeckImputation(df):
# 	print sum(df.isnull().sum(axis=0).tolist())
	# for i in range (0,df.shape[1]):
	# 	if("Unnamed" not in  df.columns[i] and "RID" not in  df.columns[i]):
	# 		print i
			
	# 		df[df.columns[i]] = df.groupby('RID')[df.columns[i]].fillna(method='ffill')
	# 		df[df.columns[i]] = df.groupby('RID')[df.columns[i]].fillna(method='bfill')
	# 		mode =  df[df.columns[i]].mode()[0]
	# 		df[df.columns[i]] = df.groupby('RID')[df.columns[i]].fillna(mode)

	# print sum(df.isnull().sum(axis=0).tolist())
	# print sum(df.isnull().sum(axis=0).tolist())
	df = df.groupby(['RID'], as_index=False).apply(lambda group: group.ffill())
	# print sum(df.isnull().sum(axis=0).tolist())
	df = df.groupby(['RID'], as_index=False).apply(lambda group: group.bfill())


	# print sum(df.isnull().sum(axis=0).tolist())
	mode = df.mode()
	df = df.fillna(mode.iloc[0])
	# print sum(df.isnull().sum(axis=0).tolist())
	# mode = mode.iloc[0]
	df = df.fillna(value = -10)
	# df.to_csv('../Finaldata/tempdf_with_Missing_Data.csv')
	# print sum(df.isnull().sum(axis=0).tolist())
	return df



# def MICE(df):
# 	imputer = skmice.MiceImputer()
# 	print sum(df.isnull().sum(axis=0).tolist()) , "abo om da "
# 	df.values, specs = imputer.transform(df.values, LinearRegression, 10)
# 	print sum(df.isnull().sum(axis=0).tolist())
# 	return df


# def KNNImputation(df,k):
# 	print ( df.shape)
# 	XY_incomplete = df.values
# 	XY_completed = KNN(k).complete(XY_incomplete)
# 	print ( XY_completed.shape)
# 	df.values = XY_completed
# 	return df



def ImputationBaseline(df):
	# print sum(df.isnull().sum(axis=0).tolist())
	tmp = df.fillna(value = -10)
	# print sum(tmp.isnull().sum(axis=0).tolist())
	return tmp


# df = read_csv('../Finaldata/df_with_Missing_Data.csv', names=None)
# HotDeckImputation(df)


