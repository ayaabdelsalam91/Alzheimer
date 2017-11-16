import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame






def expandFeature(value,Category):
	oneHot = np.zeros((Category,1))
	oneHot[int(value-1)] = 1
	return oneHot



def getDataOfIntrest(TargetName , inputfile , OuputFile, otherFeatures=None  , detailFile=None ):

	data = '../Finaldata/'+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	info = dataframe.values
	# print (data , info.shape)

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
	if(detailFile==None):
		df =DataFrame(data=dataOfIntrest, columns = cols)
	else:

		FeatureDetailsdata = '../Finaldata/'+detailFile+ '.csv'
		FeatureDetailDF = read_csv(FeatureDetailsdata, names=None)
		FeaturesDetailsInfo =  FeatureDetailDF.values



		Features=FeatureDetailDF.columns.get_loc("Feature")
		NumberOFFeatures=FeatureDetailDF.columns.get_loc("NumberOFFeatures")
		numberOfFeatures = np.ones((dataOfIntrest.shape[1],1))
		for i in range (len(cols)):
			for j in range(FeaturesDetailsInfo.shape[0]):
				if(cols[i]==FeaturesDetailsInfo[j,Features]):
					numberOfFeatures[i] = FeaturesDetailsInfo[j,NumberOFFeatures]		
		# for i in range (len (cols)):
		# 	print(cols[i] ,numberOfFeatures[i] )
		newSize=0
		for j in range(len(cols)):
				newSize+=numberOfFeatures[j,0]
		newData = np.zeros((dataOfIntrest.shape[0] , int(newSize)))

		
		#print (dataOfIntrest.shape[1] , numberOfFeatures)
		for i in range(dataOfIntrest.shape[0]):
			currentCol=0
			for j in range(dataOfIntrest.shape[1]):
				if(numberOfFeatures[j,0]==1):
					newData[i,currentCol] = dataOfIntrest[i,j]
				else:
					newData[i,currentCol:currentCol+int(numberOfFeatures[j,0])] =  expandFeature(dataOfIntrest[i,j],int(numberOfFeatures[j,0]))[:,0]
				currentCol+=int(numberOfFeatures[j,0])
		newCols = []
		for i in range (len(cols)):
			if(numberOfFeatures[i]==1):
				newCols.append(cols[i])
			else:
				for j in range (int(numberOfFeatures[i,0])):
					newCols.append(cols[i]+str(j+1))
		df =DataFrame(data=newData, columns = newCols)



	print(OuputFile , df.shape)
	df.to_csv('../Finaldata/'+OuputFile+'.csv',index=False)
	# if otherFeatures==None: 
	# 	df.to_csv('../Finaldata/temp'+TargetName+'Uni.csv',index=False)
	# else:
	# 	df.to_csv('../Finaldata/temp'+TargetName+'Multi.csv',index=False)


def ChangeFeatures(ListOfFeatures, detailFile ,  inputfile ,outfile):
	data = '../Finaldata/'+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	columns = dataframe.columns.values

	FeatureDetailsdata = '../Finaldata/'+detailFile+ '.csv'
	FeatureDetailDF = read_csv(FeatureDetailsdata, names=None)
	FeaturesDetailsInfo =  FeatureDetailDF.values


	Features=FeatureDetailDF.columns.get_loc("Feature")
	Category_1_from=FeatureDetailDF.columns.get_loc("Category_1_from")
	Category_1_to = FeatureDetailDF.columns.get_loc("Category_1_to")
	Category_2_from = FeatureDetailDF.columns.get_loc("Category_2_from")
	Category_2_to = FeatureDetailDF.columns.get_loc("Category_2_to")
	Category_3_from =FeatureDetailDF.columns.get_loc("Category_3_from")	
	Category_3_to = FeatureDetailDF.columns.get_loc("Category_3_to")
	Category_4_from  =FeatureDetailDF.columns.get_loc("Category_4_from")
	Category_4_to  =FeatureDetailDF.columns.get_loc("Category_4_to")
	min  =FeatureDetailDF.columns.get_loc("min")
	max  =FeatureDetailDF.columns.get_loc("max")

	DX=  dataframe.columns.get_loc("DX")
	DXCHANGE=  dataframe.columns.get_loc("DXCHANGE")
	DX_bl=  dataframe.columns.get_loc("DX_bl")
	for i in range(info.shape[0]):
		if(info[i,DX]==7 or info[i,DX]==9):
			info[i,DX]=1
		elif(info[i,DX]==4 or info[i,DX]==8):
			info[i,DX]=2
		elif(info[i,DX]==5 or info[i,DX]==6):
			info[i,DX]=3

		if(info[i,DXCHANGE]==7 or info[i,DXCHANGE]==9):
			info[i,DXCHANGE]=1
		elif(info[i,DXCHANGE]==4 or info[i,DXCHANGE]==8):
			info[i,DXCHANGE]=2
		elif(info[i,DXCHANGE]==5 or info[i,DXCHANGE]==6):
			info[i,DXCHANGE]=3

		if(info[i,DX_bl]==7 or info[i,DX_bl]==9):
			info[i,DX_bl]=1
		elif(info[i,DX_bl]==4 or info[i,DX_bl]==8):
			info[i,DX_bl]=2
		elif(info[i,DX_bl]==5 or info[i,DX_bl]==6):
			info[i,DX_bl]=3

	newFeatures = info
	for i in range (info.shape[1]):
		for j in range(FeaturesDetailsInfo.shape[0]):
			if(columns[i]==FeaturesDetailsInfo[j,Features]):
				# print(columns[i])
				for indx in range (info.shape[0]):
					value = info[indx,i ]
					if(not np.isnan(FeaturesDetailsInfo[j,Category_1_from])  and not np.isnan(FeaturesDetailsInfo[j,Category_1_to]) ):
						if(value<=FeaturesDetailsInfo[j,Category_1_from] and value>=FeaturesDetailsInfo[j,Category_1_to]):
							newFeatures[indx,i ] = 1
							# print(1)

					if(not np.isnan(FeaturesDetailsInfo[j,Category_2_from])  and not np.isnan(FeaturesDetailsInfo[j,Category_2_to]) ):
						if(value<=FeaturesDetailsInfo[j,Category_2_from] and value>=FeaturesDetailsInfo[j,Category_2_to]):
							newFeatures[indx,i ] = 2
							# print(2)

					if(not np.isnan(FeaturesDetailsInfo[j,Category_3_from])  and not np.isnan(FeaturesDetailsInfo[j,Category_3_to]) ):
						if(value<=FeaturesDetailsInfo[j,Category_3_from] and value>=FeaturesDetailsInfo[j,Category_3_to]):
							newFeatures[indx,i ] = 3
							# print(3)

					if(not np.isnan(FeaturesDetailsInfo[j,Category_4_from])  and not np.isnan(FeaturesDetailsInfo[j,Category_4_to]) ):
						if(value<=FeaturesDetailsInfo[j,Category_4_from] and value>=FeaturesDetailsInfo[j,Category_4_to]):
							newFeatures[indx,i ] = 4
							# print(4)							
					if(not np.isnan(FeaturesDetailsInfo[j,max])  and not np.isnan(FeaturesDetailsInfo[j,min]) ):

						newFeatures[indx,i ] = float(newFeatures[indx,i ] - FeaturesDetailsInfo[j,min])/(FeaturesDetailsInfo[j,max]- FeaturesDetailsInfo[j,min])
	

	df=DataFrame(newFeatures, columns=columns)
	print(outfile , df.shape)
	df.to_csv('../Finaldata/'+outfile+'.csv',index=False)









