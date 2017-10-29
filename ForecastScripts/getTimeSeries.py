import sklearn.metrics as sm
import numpy as np
import pandas as pd
from pandas import read_csv
import csv



def getTimeSeries(TargetName, inputfile , OuputFile, multivariate=False):
	data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/'+inputfile+ '.csv'
	# if not multivariate:
	# 	data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/temp'+TargetName+'Uni.csv'
	# else:
	# 	data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/temp'+TargetName+'Multi.csv'
	# print data
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	print(data , info.shape)

	columns = dataframe.columns.values
	NumericCols =[]

	for i in range (len(columns)):
		col = dataframe.columns.get_loc(columns[i])
		NumericCols.append(col)
	RID = dataframe.columns.get_loc("RID")
	VISCODE=  dataframe.columns.get_loc("VISCODE")
	Target=  dataframe.columns.get_loc(TargetName)

	rid = info[:,RID]
	unqRID = np.unique(rid)

	viscode = info[:,VISCODE]
	unqVISCODE = np.unique(viscode)
	NumberOfExtraFeatures = info.shape[1]-2


	if not multivariate:
		data = np.ones((unqRID.shape[0],22) )*-99999999999999

		loc_map={}

		for i in range (len(unqRID)):
			data[i,0] = unqRID[i]
			loc_map[unqRID[i]]  = i

		for i in range (info.shape[0]):
			rid= loc_map[info[i,RID]]
			data[rid,int(info[i,VISCODE]/6)+1] = info[i,Target]

	else:

		data = np.ones((unqRID.shape[0],(21*NumberOfExtraFeatures)+1))*-99999999999999

		loc_map={}

		for i in range (len(unqRID)):
			data[i,0] = unqRID[i]
			loc_map[unqRID[i]]  = i

		for i in range (info.shape[0]):
			rid= loc_map[info[i,RID]]
			for j in range (NumberOfExtraFeatures):
				index = (int(info[i,VISCODE]/6)*NumberOfExtraFeatures)+1+j
				# print i ,info[i,VISCODE] , index , columns[j+2] , info[i,NumericCols[j+2]]
				data[rid,index ] = info[i,NumericCols[j+2]]
			# 	# if(j==2):
			# 	# 	break
			# if(i==1):
			# 	break
				

	temp=  ['RID']
	for i in range (21):
		for j in range (NumberOfExtraFeatures):
			temp.append(str(i*6))
	
	df = pd.DataFrame(data,columns=temp)
	df.replace(['-99999999999999', '-99999999999999.0'], '', inplace=True)
	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/'+OuputFile+'.csv',index=False)

	# if not multivariate:
	# 	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/temp'+TargetName+'Uni2.csv',index=False)
	# else:
	# 	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/temp'+TargetName+'Multi2.csv',index=False)



# getTimeSeries("Ventricles",multivariate=True)
# getTimeSeries("Ventricles")
#getTimeSeries("ADAS13", multivariate=True)
#getTimeSeries("ADAS13")
