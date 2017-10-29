import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame
np.set_printoptions(threshold=np.nan)

path = '/Users/Tim/Desktop/Alzheimer/ForecastProcessed_data/'

# def CreateLSTMTrainingAndTestingData(TargetName,TrainRID,TestRID,multivariate=False, shift=False):
def FillinDataForForecasting(TargetName,inputfile,outputfile,multivariate=False, shift=False,Testing = False):

	data =path+inputfile+'.csv'
	#print(data)
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
				for j in range (int(1+numberOfFeatures),LSTMData.shape[1] ):
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] =  LSTMData.values[i,j-numberOfFeatures]


	else:

		shiftingValue =  np.zeros((info.shape[0],1))
		for i in range (info.shape[0]):
			last = 0
			for j in range (info.shape[1]):
				if(not np.isnan(info[i,j])):
					last = j
			shiftingValue[i] = last
		for i in range (LSTMData.shape[0]):
			#print LSTMData.values[i,0] , shiftingValue[i]
			for j in range (int(1+numberOfFeatures),shiftingValue[i] ):
				if(np.isnan(LSTMData.values[i,j])):
					LSTMData.values[i,j] =  LSTMData.values[i,j-numberOfFeatures]

	if(shift):
		for i in range (len(cols)):
			if(cols[i]=="0" or cols[i].startswith( "0." )):
					LSTMData.drop(cols[i], axis=1, inplace=True)
		LSTMData.dropna(thresh=2 , inplace=True)
	LSTMData.to_csv(path+outputfile+'.csv',index=False)




def FillinDataForForecastingOLD(TargetName,inputfile,outputfile,multivariate=False, shift=False,Testing = False , TimeSteps=6):

	data =path+inputfile+'.csv'

	dataframe = read_csv(data, names=None)
	info=dataframe.values
	#print(data , info.shape)
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
	LSTMData = DataFrame(LSTMData,columns=cols)
	LSTMData.replace(['-99999999999999', '-99999999999999.0'], np.nan, inplace=True)
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
			#print(cols)
		else:
			output[:,1:] = info[:,start-1*numberOfFeatures:-1*numberOfFeatures]
			cols = ["RID"]
			for i in range(int(start-1*numberOfFeatures) ,int(info.shape[1] -1*numberOfFeatures)):
				cols.append (dataframe.columns.values[i])
			#print(cols)
		LSTMData = DataFrame(output,columns=cols)
		if( not multivariate):
			LSTMData.fillna(method='bfill', axis=1, inplace=True)
		else:
			for i in range (LSTMData.shape[0]):
				for j in range (LSTMData.shape[1]-1 , 0 , -1 ):
					if(np.isnan(LSTMData.values[i,j])):
						LSTMData.values[i,j] = LSTMData.values[i,j+numberOfFeatures]

	LSTMData.to_csv(path+outputfile+'.csv',index=False)




def DuplicateData(DataType,TargetName , multivariate=False,shift=False,isnotDx=True):

	data = path+'/LSTM'+DataType+'Data'+TargetName+'.csv'


	dataframe = read_csv(data, names=None)
	info =dataframe.values
	#print(data , info.shape)

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
			#print("DX!")
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
					for col in range(1 , int(NumOfcols-1)):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[j+i*14,col+1]=info[i,col+numberOfFeatures + j*numberOfFeatures]-info[i,col+ j*numberOfFeatures]

					if(output[j+i*14,2:].sum()!=0):
						rowsOfIntrest.append(j+i*14)
			output = output[rowsOfIntrest]
			cols = ["RID" , "NumberInSeq"]
			for i in range(1,6):
				for j in range (int(numberOfFeatures)):
					cols.append("Feature_" +  str(j+1) +  "_diff_" +  str(i))
			cols.append("Target")
		else:
			#print("DX")
			for i in range (info.shape[0]):
				for j in range (0,14):
					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					for col in range(1 , int(NumOfcols-1)):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[j+i*14,col+1]=info[i,col+ j*numberOfFeatures]
			
			cols = ["RID" , "NumberInSeq"]
			for i in range(1,6):
				for j in range (int(numberOfFeatures)):
					cols.append("Feature_" +  str(j+1) +  "_time_" +  str(i))
			cols.append("Target")
		#print columnsname
	df = DataFrame(output , columns = cols)
	#print(df.shape)
	df.to_csv(path+DataType+'Data'+TargetName+'.csv',index=False)




def getLSTMInput(inputfile,outputfile,shift=False,Testing = False,DiffTarget=False , DiffFeatures=False):
	data = path+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info =dataframe.values
	print(data , info.shape)
	RID = dataframe.columns.get_loc("RID")

	if(not shift):
		NumberOfTimeSteps=21
	else:
		NumberOfTimeSteps=20
	numberOfFeatures =  (info.shape[1]-1)/NumberOfTimeSteps
	if(not Testing):
		val1 = info.shape[0]*(NumberOfTimeSteps-1)
		val2 = int(info.shape[1]+1-numberOfFeatures)
		print(val1,val2)
		output = np.zeros((val1,val2))
		if(not DiffTarget):
			for i in range (info.shape[0]):
				for j in range (0,(NumberOfTimeSteps-1)):
					output[j+i*(NumberOfTimeSteps-1),0]=info[i,RID]
					for col in range(1 , int((j+1)*numberOfFeatures+1)):
						output[j+i*(NumberOfTimeSteps-1),col]=info[i,col]
					output[j+i*(NumberOfTimeSteps-1),-1] = info[i,col+1]
		elif(DiffTarget):
			for i in range (info.shape[0]):
				for j in range (0,(NumberOfTimeSteps-1)):
					output[j+i*(NumberOfTimeSteps-1),0]=info[i,RID]
					for col in range(1 , int((j+1)*int(numberOfFeatures)+1)):
						output[j+i*(NumberOfTimeSteps-1),col]=info[i,col]
					output[j+i*(NumberOfTimeSteps-1),-1] = info[i,col+1]-info[i,col-int(numberOfFeatures)+1]
		cols=[]
		for i in range(int(dataframe.columns.shape[0]-numberOfFeatures)):
			cols.append(dataframe.columns.values[i])
		cols.append("Target")

	else:
		output = np.zeros((info.shape))
		for i in range (info.shape[0]):
			for j in range (info.shape[1]):
				if(not np.isnan(info[i,j])):
					output[i,j] = info[i,j]
				else:
					break
		cols=[]
		for i in range(dataframe.columns.shape[0]):
			cols.append(dataframe.columns.values[i])
	df = DataFrame(output ,columns=cols)
	#print(df.shape)
	df.to_csv(path+outputfile+'.csv',index=False)





def DuplicateDataForForcasting(inputfile,outputfile , TargetName , multivariate=False,shift=False,isnotDx=True):
	data = path+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info =dataframe.values
	#print(data , info.shape)
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
			#print("DX!")
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
					for col in range(1 , int(NumOfcols-1)):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[j+i*14,col+1]=info[i,col+numberOfFeatures + j*numberOfFeatures]-info[i,col+ j*numberOfFeatures]

					if(output[j+i*14,2:].sum()!=0):
						rowsOfIntrest.append(j+i*14)
			output = output[rowsOfIntrest]
			cols = ["RID" , "NumberInSeq"]
			for i in range(1,6):
				for j in range (int(numberOfFeatures)):
					cols.append("Feature_" +  str(j+1) +  "_diff_" +  str(i))
			cols.append("Target")
		else:
			#print "DX"
			for i in range (info.shape[0]):
				for j in range (0,14):
					output[j+i*14,0]=info[i,RID]
					output[j+i*14,1]= j+1
					for col in range(1 , int(NumOfcols-1)):
						#print  " j+i*14" , j+i*14, "diff" , diff , "col+1" , col+1 , "j+col+numberOfFeatures"  ,dataframe.columns.values[j+col+numberOfFeatures + j*numberOfFeatures] , "j+col" , dataframe.columns.values[j+col+j*numberOfFeatures]
						output[j+i*14,col+1]=info[i,col+ j*numberOfFeatures]
			
			cols = ["RID" , "NumberInSeq"]
			for i in range(1,6):
				for j in range (int(numberOfFeatures)):
					cols.append("Feature_" +  str(j+1) +  "_time_" +  str(i))
			cols.append("Target")
		#print columnsname'



	df = DataFrame(output , columns = cols)
	#print df.shape
	df.to_csv(path+outputfile+'.csv',index=False)





def getTheLastXTimeStepsForForecasting(TargetName , inputfile,outputfile , multivariate=False,shift=False,isnotDx=True ,  TimeSteps = 5):
	data = path+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info =dataframe.values
	#print data , info.shape
	RID = dataframe.columns.get_loc("RID")
	numberOfFeatures =  (info.shape[1]-1)/21
	output = np.zeros((info.shape[0],TimeSteps*numberOfFeatures+1))
	output[:,0] = info[:,0]
	start = (21- TimeSteps)*6
	start = dataframe.columns.get_loc(str(start))
	#print info[0,:]
	output[:,1:] = info[:,start:]
	cols = ["RID"]
	for i in range(start ,info.shape[1] ):

		cols.append (dataframe.columns.values[i])
	#print cols
	df = DataFrame(output , columns = cols)
	df.to_csv(path+outputfile+'.csv',index=False)

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









