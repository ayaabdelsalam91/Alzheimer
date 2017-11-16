import sklearn.metrics as sm
import numpy as np
import pandas as pd
from pandas import read_csv
import csv



def getTimeSeries(TargetName, inputfile , OuputFile, multivariate=False,CategoricalTarget=False):
	data = '../Finaldata/'+inputfile+ '.csv'
	# if not multivariate:
	# 	data = '../Finaldata/temp'+TargetName+'Uni.csv'
	# else:
	# 	data = '../Finaldata/temp'+TargetName+'Multi.csv'
	# print data
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	# print (data , info.shape)

	columns = dataframe.columns.values
	NumericCols =[]

	for i in range (len(columns)):
		col = dataframe.columns.get_loc(columns[i])
		NumericCols.append(col)
	RID = dataframe.columns.get_loc("RID")
	VISCODE=  dataframe.columns.get_loc("VISCODE")

	
	rid = info[:,RID]
	unqRID = np.unique(rid)

	viscode = info[:,VISCODE]
	unqVISCODE = np.unique(viscode)
	NumberOfExtraFeatures = info.shape[1]-2


	if not multivariate:
		data = np.ones((unqRID.shape[0],22) )*-99999999999999
		Target=  dataframe.columns.get_loc(TargetName)
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
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if(data[i,j]==-99999999999999):
				data[i,j]=np.nan

	df = pd.DataFrame(data,columns=temp)
	# df.replace(['-99999999999999', '-99999999999999.0'], '', inplace=True)
	print(OuputFile,  df.shape)
	df.to_csv('../Finaldata/'+OuputFile+'.csv',index=False)

	# if not multivariate:
	# 	df.to_csv('../Finaldata/temp'+TargetName+'Uni2.csv',index=False)
	# else:
	# 	df.to_csv('../Finaldata/temp'+TargetName+'Multi2.csv',index=False)



# getTimeSeries("Ventricles",multivariate=True)
# getTimeSeries("Ventricles")
#getTimeSeries("ADAS13", multivariate=True)
#getTimeSeries("ADAS13")
