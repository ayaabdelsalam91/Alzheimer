import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame



def getDataOfIntrest(TargetName , inputfile , OuputFile, otherFeatures=None ):

	data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/'+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	info = dataframe.values
	print(data , info.shape)

	if(TargetName=="DX"):
		DX=  dataframe.columns.get_loc("DX")
		for i in range(info.shape[0]):
			if(info[i,DX]==7 or info[i,DX]==9):
				info[i,DX]=1
			elif(info[i,DX]==4 or info[i,DX]==8):
				info[i,DX]=2
			elif(info[i,DX]==5 or info[i,DX]==6):
				info[i,DX]=3
	RID = dataframe.columns.get_loc("RID")
	VISCODE=  dataframe.columns.get_loc("VISCODE")
	if otherFeatures==None:
		dataOfIntrest = np.zeros((dataframe.shape[0] , 3))
		dataOfIntrest[:,0] = info[:,RID]
		dataOfIntrest[:,1] = info[:,VISCODE]
		if TargetName=="Ventricles":
			Ventricles=  dataframe.columns.get_loc("Ventricles")
			ICV=  dataframe.columns.get_loc("ICV")
			dataOfIntrest[:,2] = info[:,Ventricles]/info[:,ICV]
			cols = ["RID" , "VISCODE" ,"Ventricles"]

		else:
			Target=  dataframe.columns.get_loc(TargetName)
			dataOfIntrest[:,2] = info[:,Target]
			cols = ["RID" , "VISCODE" ,TargetName]
			

	else:
		dataOfIntrest = np.zeros((dataframe.shape[0] , 3+len(otherFeatures)))
		dataOfIntrest[:,0] = info[:,RID]
		dataOfIntrest[:,1] = info[:,VISCODE]
		if TargetName=="Ventricles":
			Ventricles=  dataframe.columns.get_loc("Ventricles")
			ICV=  dataframe.columns.get_loc("ICV")
			dataOfIntrest[:,2] = info[:,Ventricles]/info[:,ICV]
			cols = ["RID" , "VISCODE" ,"Ventricles"]
		else:
			Target=  dataframe.columns.get_loc(TargetName)
			dataOfIntrest[:,2] = info[:,Target]
			cols = ["RID" , "VISCODE" ,TargetName]
		for i in range(len(otherFeatures)):
			index = dataframe.columns.get_loc(otherFeatures[i])
			if TargetName=="Ventricles":
				if (otherFeatures[i]=="Ventricles_bl"):
					ICV=  dataframe.columns.get_loc("ICV")
					dataOfIntrest[:,3+i] = info[:,index]/info[:,ICV]
				else:
					max = np.max(info[:,index])
					min = np.min(info[:,index])
					diff = float(max-min)
					dataOfIntrest[:,3+i] = (info[:,index]-min)/diff
			else:
				dataOfIntrest[:,3+i] = info[:,index]
			cols.append(otherFeatures[i])

	df =DataFrame(data=dataOfIntrest, columns = cols)

	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/'+OuputFile+'.csv',index=False)
	# if otherFeatures==None: 
	# 	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/temp'+TargetName+'Uni.csv',index=False)
	# else:
	# 	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/temp'+TargetName+'Multi.csv',index=False)


# features = ["ST37SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16" , 
# "ST96SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
# "ST30SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
# "ST127SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
# "ST89SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
# "ST37SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16"]
# getDataOfIntrest("Ventricles" , features)




# def getDX(otherFeatures=None):
# 	data = '/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/Local_TrainingAfterImpHD.csv'
# 	dataframe = read_csv(data, names=None)

# 	info = dataframe.values
# 	print data , info.shape

# 	RID = dataframe.columns.get_loc("RID")
# 	VISCODE=  dataframe.columns.get_loc("VISCODE")
# 	DX=  dataframe.columns.get_loc("DX")
# 	for i in range(info.shape[0]):
# 		if(info[i,DX]==7 or info[i,DX]==9):
# 			info[i,DX]=1
# 		elif(info[i,DX]==4 or info[i,DX]==8):
# 			info[i,DX]=2
# 		elif(info[i,DX]==5 or info[i,DX]==6):
# 			info[i,DX]=3


# 	if(otherFeatures==None):
# 		dataOfIntrest = np.zeros((dataframe.shape[0] , 3))
# 		dataOfIntrest[:,0] = info[:,RID]
# 		dataOfIntrest[:,1] = info[:,VISCODE]
# 		dataOfIntrest[:,2] = info[:,DX]
# 		cols = ["RID" , "VISCODE" ,"DX"]
# 	else:
# 		dataOfIntrest = np.zeros((dataframe.shape[0] , 3+len(otherFeatures)))
# 		dataOfIntrest[:,0] = info[:,RID]
# 		dataOfIntrest[:,1] = info[:,VISCODE]

# 			Target=  dataframe.columns.get_loc("DX")
# 			dataOfIntrest[:,2] = info[:,Target]
# 			cols = ["RID" , "VISCODE" ,"DX"]
# 		for i in range(len(otherFeatures)):
# 			index = dataframe.columns.get_loc(otherFeatures[i])
# 			if TargetName=="Ventricles":
# 				if (otherFeatures[i]=="Ventricles_bl"):
# 					ICV=  dataframe.columns.get_loc("ICV")
# 					dataOfIntrest[:,3+i] = info[:,index]/info[:,ICV]
# 				else:
# 					max = np.max(info[:,index])
# 					min = np.min(info[:,index])
# 					diff = float(max-min)
# 					dataOfIntrest[:,3+i] = (info[:,index]-min)/diff
# 			else:
# 				dataOfIntrest[:,3+i] = info[:,index]
# 			cols.append(otherFeatures[i])
# 	df =DataFrame(data=dataOfIntrest, columns = cols)
# 	df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/temp'+"DX"+'Uni.csv',index=False)

			

# getDataOfIntrest("Ventricles" )


# features = ["RAVLT_immediate_bl" , 
# "RAVLT_immediate",
# "DXCHANGE",
# "MMSE",
# "MMSE_bl",
# "CDRSB_bl"]
# getDataOfIntrest("ADAS13" , features)
		

# RAVLT_immediate_bl 13.649817     28011.985
# RAVLT_immediate    12.445079     24789.614
# DXCHANGE            8.194124     15691.317
# MMSE                3.413586     10181.456
# MMSE_bl             3.630710      9491.828
# CDRSB_bl            3.108336      7283.003


# getDataOfIntrest("ADAS13" )

