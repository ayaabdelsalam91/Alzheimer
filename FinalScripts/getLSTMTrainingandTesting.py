import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame
from scipy import stats
np.set_printoptions(threshold=np.nan)



# def CreateLSTMTrainingAndTestingData(TargetName,TrainRID,TestRID,multivariate=False, shift=False):
def FillinDataForForecasting(TargetName,inputfile,outputfile,multivariate=False, shift=False,Testing = False):
	data ='../Finaldata/'+inputfile+'.csv'
	# print (data)
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	cols =  dataframe.columns.values
	RID = dataframe.columns.get_loc("RID")
	numberOfFeatures =  (info.shape[1]-1)/21
	LSTMData =  DataFrame(info , columns=cols)
	
	if(not Testing):
		if( not multivariate):
			LSTMData.fillna(method='ffill', axis=1, inplace=True)
		else:
			for i in range (LSTMData.shape[0]):
				for j in range (int(1+numberOfFeatures),int(LSTMData.shape[1]) ):
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] =  LSTMData.values[i,int(j-numberOfFeatures)]

	else:
		# count=0

		shiftingValue =  np.zeros((info.shape[0],1))
		for i in range (info.shape[0]):
			last = 0
			for j in range (1,info.shape[1]):
				if(not np.isnan(info[i,j])):
					last = j
			shiftingValue[i] = last
			# if(shiftingValue[i]==1):
			# 	shiftingValue[i]+=1

		if(not multivariate):
			for i in range (LSTMData.shape[0]):
				for j in range (2,int(shiftingValue[i] )):
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] =  LSTMData.values[i,j-1]
			for i in range (LSTMData.shape[0]):
				for j in range (int(shiftingValue[i])-1 , 0 ,-1):
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] =  LSTMData.values[i,j+1]

		else:
			for i in range (LSTMData.shape[0]):
				for j in range (int(1+numberOfFeatures),int(shiftingValue[i] )):
					# print (j , numberOfFeatures)
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] =  LSTMData.values[i,int(j-numberOfFeatures)]

			for i in range (LSTMData.shape[0]):
				for j in range (int(shiftingValue[i]), 0,-1):
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] =  LSTMData.values[i,int(j+numberOfFeatures)]

	if(shift):
		# print("LOL!")
		# if(not multivariate):
		# 	for i in range (LSTMData.shape[0]):
		# 		if(np.isnan(LSTMData.values[i,2]) and not np.isnan(LSTMData.values[i,1])):
		# 			LSTMData.values[i,2]=LSTMData.values[i,1]
		# 			print(LSTMData.values[i,0])
		# else:
		# 	for i in range (LSTMData.shape[0]):
		# 		if(np.isnan(LSTMData.values[i,1+numberOfFeatures]) and not np.isnan(LSTMData.values[i,1])):
		# 			print(LSTMData.values[i,0])
		# 			LSTMData.values[i,1+numberOfFeatures:1+numberOfFeatures+numberOfFeatures]=LSTMData.values[i,1:1+numberOfFeatures]


		for i in range (len(cols)):
			if(cols[i]=="0" or cols[i].startswith( "0." )):
					LSTMData.drop(cols[i], axis=1, inplace=True)
		# LSTMData.dropna(thresh=2 , inplace=True)
	print (outputfile ,  LSTMData.shape)
	LSTMData.to_csv('../Finaldata/'+outputfile+'.csv',index=False)



def getValueOfIntrest(Listinputfile ,  detailFile = None , Orignal=None):
	valuesOfIntrest=[]
	isBaseLine =[] 


	if(detailFile==None):
		for inputfile in Listinputfile:
			data ='../Finaldata/'+inputfile+'.csv'

			dataframe = read_csv(data, names=None)
			info=dataframe.values
			
			for i in range(info.shape[1]):
				if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE"  ):
					unq = np.unique(info[:,i])

					if(len(unq)>50):
						
						valuesOfIntrest.append(np.mean(info[:,i]))
					else:
						valuesOfIntrest.append(stats.mode(info[:,i])[0].flatten()[0])
					# print (dataframe.columns.values[i])
					if("bl" in  dataframe.columns.values[i] ):
						isBaseLine.append(1)
					else:
						isBaseLine.append(0)

			
	else:


		detailFiledata ='../Finaldata/'+detailFile+'.csv'
		detailFileDF = read_csv(detailFiledata, names=None)
		detailFileinfo=detailFileDF.values
		Features=detailFileDF.columns.get_loc("Feature")
		NumberOFFeatures =detailFileDF.columns.get_loc( "NumberOFFeatures")

		OrignalFiledata ='../Finaldata/'+Orignal+'.csv'
		OrignalFileDF = read_csv(OrignalFiledata, names=None)
		Orignalinfo = OrignalFileDF.values
		vistedFeatures = np.zeros((detailFileinfo.shape[0],1))
		for inputfile in Listinputfile:
			data ='../Finaldata/'+inputfile+'.csv'
			dataframe = read_csv(data, names=None)
			info=dataframe.values
			cols = dataframe.columns.values

			test=[]
			for i in range(info.shape[1]):
				if(cols[i]!="RID" and cols[i]!="VISCODE"  ):
					if("bl" in cols[i] ):
						isBaseLine.append(1)
					else:
						isBaseLine.append(0)
					unq = np.unique(info[:,i])

					if(len(unq)>50):
						
						valuesOfIntrest.append(np.mean(info[:,i]))
					else:
						flag=0
						for j in range(detailFileinfo.shape[0]):
							if(vistedFeatures[j]==0):
								if(detailFileinfo[j,Features] in cols[i]):
									flag=1
									vistedFeatures[j]=1
									OneHot =np.zeros((detailFileinfo[j,NumberOFFeatures],1))
									for index in range(OrignalFileDF.shape[1]):
										if(OrignalFileDF.columns.values[index] == detailFileinfo[j,Features]):

											value = stats.mode(Orignalinfo[:,index])[0].flatten()[0]
											
											break
									OneHot[int(value-1)]=1
									#print(detailFileinfo[j,Features] , value )
									for v in range(OneHot.shape[0]):
										test.append(OneHot[v,0])
										#print (OneHot[v,0])
										valuesOfIntrest.append(OneHot[v,0])
									
									i=+NumberOFFeatures
									break
							elif(vistedFeatures[j]==1 and detailFileinfo[j,Features] in cols[i]):
								flag=1
						if(flag==0):
							print(cols[i])
							valuesOfIntrest.append(stats.mode(info[:,i])[0].flatten()[0])
					# print (cols[i])







	return valuesOfIntrest , isBaseLine 




def FillinDataForForecastingOLD(TargetName,inputfile,outputfile,multivariate=False, shift=False,Testing = False , TimeSteps=6):

	data ='../Finaldata/'+inputfile+'.csv'

	dataframe = read_csv(data, names=None)
	info=dataframe.values
	print (data , info.shape)
	cols =  dataframe.columns.values
	RID = dataframe.columns.get_loc("RID")
	numberOfFeatures =  (info.shape[1]-1)/21
	shiftingValue =  np.zeros((info.shape[0],1))
	for i in range (info.shape[0]):
		last = 0
		for j in range (info.shape[1]):
			if(not np.isnan(info[i,j])):
				last = j
		shiftingValue[i] = (info.shape[1] - last -1)/numberOfFeatures
		
	
	LSTMData =  np.ones((info.shape))*-99999999999999
	for i in range (info.shape[0]):
		LSTMData[i,0] = info[i,0]
		for j in range (1, info.shape[1]):
			# if (shiftingValue[i]>0):
			# 	print info[i,0] , shiftingValue[i] , int(j+shiftingValue[i]*numberOfFeatures) , j  ,"hh"
			if(not np.isnan(info[i,j])):
				LSTMData[i,int(j+shiftingValue[i,0]*numberOfFeatures)] = info[i,j]
	for i in range(LSTMData.shape[0]):
		for j in range(LSTMData.shape[1]):
			if(LSTMData[i,j]==-99999999999999):
				print("in")
				LSTMData[i,j]=np.nan
	LSTMData = DataFrame(LSTMData,columns=cols)
	# LSTMData.replace(['-99999999999999', '-99999999999999.0'], np.nan, inplace=True)

	if(not Testing):

		if( not multivariate):
			LSTMData.fillna(method='bfill', axis=1, inplace=True)
		else:
			for i in range (LSTMData.shape[0]):
				for j in range (LSTMData.shape[1]-1 , 0 , -1 ):
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] = LSTMData.values[i,j+numberOfFeatures]

		if(shift):
			for i in range (len(cols)):
				if(cols[i]=="0" or cols[i].startswith( "0." )):
						LSTMData.drop(cols[i], axis=1, inplace=True)
			LSTMData.dropna(thresh=2 , inplace=True)
		
	else:
		info =LSTMData.values
		output = np.zeros((info.shape[0],TimeSteps*numberOfFeatures+1))
		output[:,0] = info[:,0]
		start = (21- TimeSteps)*6
		start = dataframe.columns.get_loc(str(start))
		if(not shift):
			output[:,1:] = info[:,start:]
			cols = ["RID"]
			for i in range(start ,info.shape[1] ):
				cols.append (dataframe.columns.values[i])
			print (cols)
		else:
			output[:,1:] = info[:,start-1*numberOfFeatures:-1*numberOfFeatures]
			cols = ["RID"]
			for i in range(start-1*numberOfFeatures ,info.shape[1] -1*numberOfFeatures):
				cols.append (dataframe.columns.values[i])
			print (cols)
		LSTMData = DataFrame(output,columns=cols)
		if( not multivariate):
			LSTMData.fillna(method='bfill', axis=1, inplace=True)
		else:
			for i in range (LSTMData.shape[0]):
				for j in range (LSTMData.shape[1]-1 , 0 , -1 ):
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] = LSTMData.values[i,j+numberOfFeatures]

	LSTMData.to_csv('../Finaldata/'+outputfile+'.csv',index=False)




def DuplicateData(DataType,TargetName , multivariate=False,shift=False,isnotDx=True):

	data = '../Finaldata/LSTM'+DataType+'Data'+TargetName+'.csv'


	dataframe = read_csv(data, names=None)
	info =dataframe.values
	print (data , info.shape)

	RID = dataframe.columns.get_loc("RID")
	if( not multivariate):
		output = np.zeros((info.shape[0]*14,8))
		rowsOfIntrest =[]
		if(isnotDx):

			for i in range (info.shape[0]):
				for j in range (0,14):

					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					output[j+i*14,2]=info[i,j+2]-info[i,j+1]
					output[j+i*14,3]=info[i,j+3]-info[i,j+2]
					output[j+i*14,4]=info[i,j+4]-info[i,j+3]
					output[j+i*14,5]=info[i,j+5]-info[i,j+4]
					output[j+i*14,6]=info[i,j+6]-info[i,j+5]
					output[j+i*14,7]=info[i,j+7]-info[i,j+6]
					if(not (output[j+i*14,2]==0 and output[j+i*14,3]==0 and output[j+i*14,4]==0 and output[j+i*14,5]==0 and output[j+i*14,6]==0 and output[j+i*14,7]==0 )):
						rowsOfIntrest.append(j+i*14)

			output = output[rowsOfIntrest]
			cols = ["RID" , "NumberInSeq" , "diff1", "diff2", "diff3", "diff4", "diff5" , "Target"]
		else:
			print ("DX!")
			for i in range (info.shape[0]):
				for j in range (0,14):

					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					output[j+i*14,2]=info[i,j+1]
					output[j+i*14,3]=info[i,j+2]
					output[j+i*14,4]=info[i,j+3]
					output[j+i*14,5]=info[i,j+4]
					output[j+i*14,6]=info[i,j+5]
					output[j+i*14,7]=info[i,j+6]

					if(not (output[j+i*14,2]==0 and output[j+i*14,3]==0 and output[j+i*14,4]==0 and output[j+i*14,5]==0 and output[j+i*14,6]==0 and output[j+i*14,7]==0 )):
						rowsOfIntrest.append(j+i*14)
			cols = ["RID" , "NumberInSeq" , "time1", "time2", "time3", "time4", "time5" , "Target"]
	else:
		if (not shift):
			numberOfFeatures =  (info.shape[1]-1)/21
		else:
			numberOfFeatures =  (info.shape[1]-1)/20
		# print numberOfFeatures
		NumOfcols  = 3+ 5*numberOfFeatures
		output = np.zeros((info.shape[0]*14,NumOfcols))
		# print cols , output.shape
		rowsOfIntrest =[]
		if(isnotDx):
			for i in range (info.shape[0]):
				for j in range (0,14):
					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					for col in range(1 , NumOfcols-1):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[j+i*14,col+1]=info[i,col+numberOfFeatures + j*numberOfFeatures]-info[i,col+ j*numberOfFeatures]

					if(output[j+i*14,2:].sum()!=0):
						rowsOfIntrest.append(j+i*14)
			output = output[rowsOfIntrest]
			cols = ["RID" , "NumberInSeq"]
			for i in range(1,6):
				for j in range (numberOfFeatures):
					cols.append("Feature_" +  str(j+1) +  "_diff_" +  str(i))
			cols.append("Target")
		else:
			print ("DX")
			for i in range (info.shape[0]):
				for j in range (0,14):
					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					for col in range(1 , NumOfcols-1):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[j+i*14,col+1]=info[i,col+ j*numberOfFeatures]
			
			cols = ["RID" , "NumberInSeq"]
			for i in range(1,6):
				for j in range (numberOfFeatures):
					cols.append("Feature_" +  str(j+1) +  "_time_" +  str(i))
			cols.append("Target")
		#print columnsname
	df = DataFrame(output , columns = cols)

	df.to_csv('../Finaldata/LSTMParsed'+DataType+'Data'+TargetName+'.csv',index=False)




def getLSTMInput(inputfile,outputfile,shift=False,Testing = False,DiffTarget=False , DiffFeatures=False,hasTarget=False,Categories=None):
	data = '../Finaldata/'+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info =dataframe.values
	# print data , info.shape
	RID = dataframe.columns.get_loc("RID")

	if(not shift):
		NumberOfTimeSteps=21
	else:
		NumberOfTimeSteps=20
	numberOfFeatures =  (info.shape[1]-1)/NumberOfTimeSteps
	if(not Testing):
		if Categories==None:
			output = np.zeros((int(info.shape[0]*(NumberOfTimeSteps-1)),int(info.shape[1]+1-numberOfFeatures)))
			if(not DiffTarget):
				for i in range (info.shape[0]):
					for j in range (0,int(NumberOfTimeSteps-1)):
						output[j+i*(NumberOfTimeSteps-1),0]=info[i,RID]
						for col in range(1 , int((j+1)*numberOfFeatures+1)):
							output[j+i*(NumberOfTimeSteps-1),col]=info[i,col]
						output[j+i*(NumberOfTimeSteps-1),-1] = info[i,col+1]
			elif(DiffTarget):
				for i in range (info.shape[0]):
					for j in range (0,int(NumberOfTimeSteps-1)):
						output[j+i*(NumberOfTimeSteps-1),0]=info[i,RID]
						for col in range(1 , int((j+1)*numberOfFeatures+1)):
							output[j+i*(NumberOfTimeSteps-1),col]=info[i,col]
						output[j+i*(NumberOfTimeSteps-1),-1] = info[i,col+1]-info[i,col-numberOfFeatures+1]
			cols=[]
			for i in range(int(dataframe.columns.shape[0]-numberOfFeatures)):
				cols.append(dataframe.columns.values[i])
			cols.append("Target")
		else:
			output = np.zeros((int(info.shape[0]*(NumberOfTimeSteps-1)),int(info.shape[1]+Categories-numberOfFeatures)))
			if(not DiffTarget):
				for i in range (info.shape[0]):
					for j in range (0,int(NumberOfTimeSteps-1)):
						output[j+i*(NumberOfTimeSteps-1),0]=info[i,RID]
						for col in range(1 , int((j+1)*numberOfFeatures+1)):
							output[j+i*(NumberOfTimeSteps-1),col]=info[i,col]
						output[j+i*(NumberOfTimeSteps-1),-int(Categories):] = info[i,col+1:col+1+Categories]
			cols=[]
			for i in range(int(dataframe.columns.shape[0]-numberOfFeatures)):
				cols.append(dataframe.columns.values[i])
			for i in range (Categories):
				cols.append("Target" + str(i))

	else:
		if(not hasTarget):
			output = np.zeros((info.shape))
			for i in range (info.shape[0]):
				for j in range (info.shape[1]):
					if(not np.isnan(info[i,j])):
						output[i,j] = info[i,j]
					# else:
					# 	break
			cols=[]
			for i in range(dataframe.columns.shape[0]):
				cols.append(dataframe.columns.values[i])
		else:
			output = np.zeros((info.shape[0],info.shape[1]+1-numberOfFeatures))
			Targets =  np.zeros((info.shape[0],1))

			shiftingValue =  np.zeros((info.shape[0],1))
			for i in range (info.shape[0]):
				for j in range (1,info.shape[1],numberOfFeatures):
					if(not np.isnan(info[i,j])):
						Targets[i] = info[i,j]
						shiftingValue[i] = j
					else:
						break
			if(not DiffTarget):
				for i in range (info.shape[0]):
					for j in range(shiftingValue[i]):
						output[i,j]=info[i,j]
				output[:,-1]= Targets[:,0]

			cols=[]
			for i in range(dataframe.columns.shape[0]-numberOfFeatures):
				cols.append(dataframe.columns.values[i])
			cols.append("Target")

	# rowsOfIntrest=[]
	# for i in range (output.shape[0]):
	# 	if(np.sum(output[i,1:-1])!=0):
	# 		rowsOfIntrest.append(i)

	# output= output[rowsOfIntrest]
	# print output.shape, len(cols)
	df = DataFrame(output ,columns=cols)
	print (outputfile ,  df.shape)

	df.to_csv('../Finaldata/'+outputfile+'.csv',index=False)




def getLSTMinXTimeSteps(inputfile,outputfile , multivariate=False, shift=False,Testing=False):
	data = '../Finaldata/'+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info =dataframe.values
	if(not shift):
		NumberOfTimeSteps=21
	else:
		NumberOfTimeSteps=20


	RID = dataframe.columns.get_loc("RID")
	if(not Testing):
		if( not multivariate):
			output = np.zeros((info.shape[0]*NumberOfTimeSteps,3))
			rowsOfIntrest =[]
			for i in range (info.shape[0]):
				for j in range (0,NumberOfTimeSteps):

					output[j+i*NumberOfTimeSteps,0]=info[i,RID]
					output[j+i*NumberOfTimeSteps,1]=info[i,j+1]
					utput[j+i*NumberOfTimeSteps,2]=info[i,j+2]

					if(not (output[j+i*NumberOfTimeSteps,2]==0 and output[j+i*NumberOfTimeSteps,3]==0 )):
						rowsOfIntrest.append(j+i*NumberOfTimeSteps)

			output = output[rowsOfIntrest]
		else:
			if (not shift):
				numberOfFeatures =  (info.shape[1]-1)/21
			else:
				numberOfFeatures =  (info.shape[1]-1)/20
			NumOfcols  = 2+numberOfFeatures
			output = np.zeros((int(info.shape[0]*NumberOfTimeSteps),int(NumOfcols)))

			rowsOfIntrest =[]

			for i in range (info.shape[0]):
				for j in range (0,NumberOfTimeSteps-1):
					output[j+i*NumberOfTimeSteps,0]=info[i,RID]
					for col in range(1 ,int( NumOfcols-1)):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[int(j+i*NumberOfTimeSteps),col]=info[i,int(col+ j*numberOfFeatures)]
					output[int(j+i*NumberOfTimeSteps),-1]=info[i,int(col+1+ j*numberOfFeatures)]
					if(output[int(j+i*NumberOfTimeSteps),1:].sum()!=0):
						rowsOfIntrest.append(j+i*NumberOfTimeSteps)
			output = output[rowsOfIntrest]
			cols = ["RID"]
			for j in range (int(numberOfFeatures)):
				cols.append("Feature_" +  str(j+1))
			cols.append("Target")
	else:
		if( not multivariate):
			output = np.zeros((info.shape[0],2))
			for i in range (info.shape[0]):
				output[i,0]=info[i,RID]
				count=0
				for j in range (1,NumberOfTimeSteps):
					if(not np.isnan(info[i,j])):
						count+=1
					else:
						break
				output[i,1] = info[i,count]


		else:
			if (not shift):
				numberOfFeatures =  (info.shape[1]-1)/21
			else:
				numberOfFeatures =  (info.shape[1]-1)/20
			NumOfcols  = 1+numberOfFeatures
			output = np.zeros((info.shape[0],int(NumOfcols)))
			for i in range (info.shape[0]):
				output[i,0]=info[i,RID]
				count=0
				for j in range(1,info.shape[1],int(numberOfFeatures)):
					if(not np.isnan(info[i,j])):
						count=j
					else:
						break

				idx = 1
				for index in range(count,int(count+numberOfFeatures)):
					
					output[i,idx] = info[i,index]
					idx+=1



		# else:
		# 	if (not shift):
		# 		numberOfFeatures =  (info.shape[1]-1)/21
		# 	else:
		# 		numberOfFeatures =  (info.shape[1]-1)/20
		# 	NumOfcols  = 1+numberOfFeatures
		# 	output= np.zeros((int(info.shape[0]*NumberOfTimeSteps),int(NumOfcols)))

		# 	rowsOfIntrest =[]

		# 	for i in range (info.shape[0]):
		# 		for j in range (0,NumberOfTimeSteps-1):
		# 			output[j+i*NumberOfTimeSteps,0]=info[i,RID]
		# 			for col in range(1 ,int( NumOfcols-1)):
		# 				#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
		# 				output[int(j+i*NumberOfTimeSteps),col]=info[i,int(col+ j*numberOfFeatures)]

		# 			if(not np.isnan(output[int(j+i*NumberOfTimeSteps),1])):
		# 				rowsOfIntrest.append(j+i*NumberOfTimeSteps)
		# 	output = output[rowsOfIntrest]
			
		cols = ["RID"]
		for j in range (int(numberOfFeatures)):
			cols.append("Feature_" +  str(j+1))









	df = DataFrame(output , columns = cols)
	print (outputfile ,  df.shape)


	df.to_csv('../Finaldata/'+outputfile+'.csv',index=False)







def DuplicateDataForForcasting(inputfile,outputfile , TargetName , multivariate=False,shift=False,isnotDx=True):
	data = '../Finaldata/'+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info =dataframe.values
	# print data , info.shape
	RID = dataframe.columns.get_loc("RID")
	if( not multivariate):
		output = np.zeros((info.shape[0]*14,8))
		rowsOfIntrest =[]
		if(isnotDx):

			for i in range (info.shape[0]):
				for j in range (0,14):

					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					output[j+i*14,2]=info[i,j+2]-info[i,j+1]
					output[j+i*14,3]=info[i,j+3]-info[i,j+2]
					output[j+i*14,4]=info[i,j+4]-info[i,j+3]
					output[j+i*14,5]=info[i,j+5]-info[i,j+4]
					output[j+i*14,6]=info[i,j+6]-info[i,j+5]
					output[j+i*14,7]=info[i,j+7]-info[i,j+6]
					if(not (output[j+i*14,2]==0 and output[j+i*14,3]==0 and output[j+i*14,4]==0 and output[j+i*14,5]==0 and output[j+i*14,6]==0 and output[j+i*14,7]==0 )):
						rowsOfIntrest.append(j+i*14)

			output = output[rowsOfIntrest]
			cols = ["RID" , "NumberInSeq" , "diff1", "diff2", "diff3", "diff4", "diff5" , "Target"]
		else:
			# print "DX!"
			for i in range (info.shape[0]):
				for j in range (0,14):

					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					output[j+i*14,2]=info[i,j+1]
					output[j+i*14,3]=info[i,j+2]
					output[j+i*14,4]=info[i,j+3]
					output[j+i*14,5]=info[i,j+4]
					output[j+i*14,6]=info[i,j+5]
					output[j+i*14,7]=info[i,j+6]

					if(not (output[j+i*14,2]==0 and output[j+i*14,3]==0 and output[j+i*14,4]==0 and output[j+i*14,5]==0 and output[j+i*14,6]==0 and output[j+i*14,7]==0 )):
						rowsOfIntrest.append(j+i*14)
			cols = ["RID" , "NumberInSeq" , "time1", "time2", "time3", "time4", "time5" , "Target"]
	else:
		if (not shift):
			numberOfFeatures =  (info.shape[1]-1)/21
		else:
			numberOfFeatures =  (info.shape[1]-1)/20
		# print numberOfFeatures
		NumOfcols  = 3+ 5*numberOfFeatures
		output = np.zeros((info.shape[0]*14,NumOfcols))
		# print cols , output.shape
		rowsOfIntrest =[]
		if(isnotDx):
			for i in range (info.shape[0]):
				for j in range (0,14):
					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					for col in range(1 , NumOfcols-1):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[j+i*14,col+1]=info[i,col+numberOfFeatures + j*numberOfFeatures]-info[i,col+ j*numberOfFeatures]

					if(output[j+i*14,2:].sum()!=0):
						rowsOfIntrest.append(j+i*14)
			output = output[rowsOfIntrest]
			cols = ["RID" , "NumberInSeq"]
			for i in range(1,6):
				for j in range (numberOfFeatures):
					cols.append("Feature_" +  str(j+1) +  "_diff_" +  str(i))
			cols.append("Target")
		else:
			# print "DX"
			for i in range (info.shape[0]):
				for j in range (0,14):
					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					for col in range(1 , NumOfcols-1):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[j+i*14,col+1]=info[i,col+ j*numberOfFeatures]
			
			cols = ["RID" , "NumberInSeq"]
			for i in range(1,6):
				for j in range (numberOfFeatures):
					cols.append("Feature_" +  str(j+1) +  "_time_" +  str(i))
			cols.append("Target")
		#print columnsname'



	df = DataFrame(output , columns = cols)
	# print df.shape
	df.to_csv('../Finaldata/'+outputfile+'.csv',index=False)




def getTheLastXTimeStepsForForecasting(TargetName , inputfile,outputfile , multivariate=False,shift=False,isnotDx=True ,  TimeSteps = 1):
	data = '../Finaldata/'+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info =dataframe.values
	# print data , info.shape
	RID = dataframe.columns.get_loc("RID")
	numberOfFeatures =  (info.shape[1]-1)/21
	output = np.zeros((info.shape[0],TimeSteps*numberOfFeatures+1))
	output[:,0] = info[:,0]
	start = (21- TimeSteps)*6
	start = dataframe.columns.get_loc(str(start))
	# print info[0,:]
	output[:,1:] = info[:,start:]
	cols = ["RID"]
	for i in range(start ,info.shape[1] ):

		cols.append (dataframe.columns.values[i])
	# print cols
	df = DataFrame(output , columns = cols)
	df.to_csv('../Finaldata/'+outputfile+'.csv',index=False)

	# if( not multivariate):
	# 	rowsOfIntrest =[]
	# 	if(isnotDx):
	# 		for i in range (info.shape[0]):
	# 			for j in range (0,14):

	# 				output[j+i*14,0]=info[i,RID]
	# 				output[j+i*14,1]= j+1
	# 				output[j+i*14,2]=info[i,j+2]-info[i,j+1]
	# 				output[j+i*14,3]=info[i,j+3]-info[i,j+2]
	# 				output[j+i*14,4]=info[i,j+4]-info[i,j+3]
	# 				output[j+i*14,5]=info[i,j+5]-info[i,j+4]
	# 				output[j+i*14,6]=info[i,j+6]-info[i,j+5]
	# 				output[j+i*14,7]=info[i,j+7]-info[i,j+6]
	# 				if(not (output[j+i*14,2]==0 and output[j+i*14,3]==0 and output[j+i*14,4]==0 and output[j+i*14,5]==0 and output[j+i*14,6]==0 and output[j+i*14,7]==0 )):
	# 					rowsOfIntrest.append(j+i*14)

	# 		output = output[rowsOfIntrest]
	# 		cols = ["RID" , "NumberInSeq" , "diff1", "diff2", "diff3", "diff4", "diff5" , "Target"]









