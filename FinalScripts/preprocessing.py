import sklearn.metrics as sm
import numpy as np
from pandas import read_csv , DataFrame
from sklearn.model_selection import train_test_split
from sklearn import svm
import csv





def countNumberOf3(df):
	D1 = df.columns.get_loc("D1")
	val=df.values
	count=0
	for i in range(val.shape[0]):
		if(val[i,D1]==3):
			count+=1
	return count




def getCols(index , columns ,processedinfo , text=True):

	for i in range (len(index)):

		if int(index[i]) == -4:
			return columns
		else:
			if text:
				columns.append(processedinfo.columns[int(index[i])])
			else:
				columns.append(int(index[i]))
	return columns


def FillWithBaseline(df,col):
	col = set(col)
	for i in range (0,df.shape[1]):
		if i in col:

			df[df.columns[i]] = df.groupby('RID')[df.columns[i]].fillna(method='ffill')
	return df

def FillAppearOnce(df,col):
	col = set(col)

	for i in range (0,df.shape[1]):
		if i in col:

			df[df.columns[i]] = df.groupby('RID')[df.columns[i]].fillna(method='ffill').fillna(method='bfill')
	return df


def isSameIDandYearBefore (first ,  second):
	if(first[0]==second[0]):
		# print  first[1] ,  int(first[1])%6 
		# print second[1] , int(second[1])%6
		if((second[1]/6)- (first[1]/6)==1):
			# print first , second
			return True
	else:
		return False

def FillAppearYearly(df,col ,wasnan ):
	count = 0
	col = set(col)
	info_v = df.values
	for j in range (0,df.shape[1]):
		# print "in J " , j
		if j in col:
			for i in range (1,df.shape[0]):
				# print "in i " , i
				if(np.isnan(info_v[i,j])):
					if(wasnan[i-1,j]==0):
					# print "NAN"
						if isSameIDandYearBefore(info_v[i-1,0:2],info_v[i,0:2]) :
						# print "LOLOLOLOLOLOLOLOLOY"
							count = count+1
							info_v[i,j] = info_v[i-1,j]
							wasnan[i,j]=1
			# 		else:
			# 			print "no3"
			# 	else:
			# 		print "no2"
			# else:
			# 	print "no1"
	# print count
	df = DataFrame(data = info_v , columns=df.columns.values)
	return df	





def Preprocessing(infile ,  Outfile):
	data = '../Finaldata/'+infile+'.csv'
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	data = '../Finaldata/FeaturesToChange.csv'
	Features = read_csv(data, names=None)
	Features=Features.values
	processedinfo = dataframe
	print ( countNumberOf3(processedinfo))
	D1 = processedinfo.columns.get_loc("D1")
	VISCODE = processedinfo.columns.get_loc("VISCODE")
	processedinfo_v= processedinfo.values
	for i in range (processedinfo.shape[0]):
		if(processedinfo_v[i,VISCODE]==3 and processedinfo_v[i,D1]==3 ):
			processedinfo_v[i,VISCODE]=6


	processedinfo=DataFrame(data= processedinfo_v , columns = processedinfo.columns.values)
	processedinfo = processedinfo[processedinfo['VISCODE']!=3]
	wasnan = np.zeros(processedinfo.shape)


	columns =[]
	# columns=["D1" , "D2"]
	columnsToBeDeleted = getCols(Features[:,0], columns ,processedinfo)
	print ( countNumberOf3(processedinfo))
	columns =[]
	BaselineColumns = getCols(Features[:,1], columns ,processedinfo,text=False)
	print (countNumberOf3(processedinfo))
	columns =[]
	OnceColumns = getCols(Features[:,2], columns ,processedinfo , text=False)
	print ( countNumberOf3(processedinfo))
	columns =[]
	YearlyColumns = getCols(Features[:,3], columns ,processedinfo, text=False)
	print ( countNumberOf3(processedinfo))

	print (len(columnsToBeDeleted) , len(BaselineColumns) , len(OnceColumns) , len(YearlyColumns))
	processedinfo = processedinfo.sort(['RID', 'VISCODE'])
	
	start =  sum(processedinfo.isnull().sum(axis=0).tolist())
	print ( countNumberOf3(processedinfo))
	print (start)
	processedinfo = FillWithBaseline(processedinfo,BaselineColumns)
	print ( countNumberOf3(processedinfo))
	one =  sum(processedinfo.isnull().sum(axis=0).tolist())
	print (one)
	processedinfo = FillAppearOnce(processedinfo,OnceColumns)
	two =  sum(processedinfo.isnull().sum(axis=0).tolist())
	print ( countNumberOf3(processedinfo))
	print ( two)
	processedinfo = FillAppearYearly(processedinfo,YearlyColumns, wasnan)
	three =  sum(processedinfo.isnull().sum(axis=0).tolist())
	print ( countNumberOf3(processedinfo))
	print ( three)



	#Delete columns I know are trash
	for i in range(len(columnsToBeDeleted)):
		processedinfo.drop(columnsToBeDeleted[i] , axis=1,inplace=True)

	print ( countNumberOf3(processedinfo))
	four =  sum(processedinfo.isnull().sum(axis=0).tolist())
	print (four)


	processedinfo_v= processedinfo.values
	df= DataFrame(data= processedinfo_v , columns = processedinfo.columns.values)
	print(df.shape)
	processedinfo.to_csv('../Finaldata/'+Outfile+'.csv',index =False)

