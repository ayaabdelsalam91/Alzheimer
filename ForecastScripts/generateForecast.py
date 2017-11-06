import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame , to_datetime
from datetime import date, datetime
from dateutil.relativedelta import relativedelta


data ='/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/ToForecast.csv'
dataframe = read_csv(data, names=None)
dataframe['EXAMDATE'] = to_datetime(dataframe['EXAMDATE'])
info=dataframe.values

dataDX ='/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/DXLeaderboradOutputPersistence.csv'
dataframeDX = read_csv(dataDX, names=None)
infoDX=dataframeDX.values

dataADAS13 ='/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/ADAS13LeaderboradOutputPersistence.csv'
dataframeADAS13 = read_csv(dataADAS13, names=None)
infoADAS13=dataframeADAS13.values

dataVentricles ='/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/VentriclesLeaderboradOutputPersistence.csv'
dataframeVentricles = read_csv(dataVentricles, names=None)
infoVentricles=dataframeVentricles.values

LeaderBoradTemplate ='/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/TADPOLE_Submission_Leaderboard_TeamName.csv'
LeaderBoradTemplateDF = read_csv(LeaderBoradTemplate, names=None)



dataDXLB ='/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/DXFeaturesForecastLeaderBoard.csv'
dataframeDXLB = read_csv(dataDXLB, names=None)
infoDXLB=dataframeDXLB.values

dataADAS13LB ='/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/ADAS13FeaturesForecastLeaderBoard.csv'
dataframeADAS13LB = read_csv(dataADAS13LB, names=None)
infoADAS13LB=dataframeADAS13LB.values

dataVentriclesLB ='/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/VentriclesFeaturesForecastLeaderBoard.csv'
dataframeVentriclesLB = read_csv(dataVentriclesLB, names=None)
infoVentriclesLB=dataframeVentriclesLB.values




RID = dataframe.columns.get_loc("RID")
EXAMDATE=  dataframe.columns.get_loc("EXAMDATE")
VISCODE = dataframe.columns.get_loc("VISCODE")



unqRID = np.unique(info[:,RID])
latest=np.zeros((len(unqRID), 2),dtype=info.dtype)


base =datetime(1000, 1, 1)
for i in range (len(unqRID)):
	latest[i,1] = base


for i in range (info.shape[0]):
	for j in range(len(unqRID)):
		if(unqRID[j]==info[i,RID]):
			if(latest[j,1]<info[i,EXAMDATE]):
				latest[j,0] = info[i,RID]
				latest[j,1] = datetime(info[i,EXAMDATE].year ,  info[i,EXAMDATE].month,1)


ListOFMonth = np.zeros((infoDX.shape[0]*6,2) ,dtype=info.dtype)

for i in range(latest.shape[0]):
	date = latest[i,1]+ relativedelta(months=+6)
	for j in range (300):
		ListOFMonth[i*300+j,0]=latest[i,0] 
		ListOFMonth[i*300+j,1]=date
		date += relativedelta(months=+1)




FullDX = np.zeros((infoDX.shape[0]*6,infoDX.shape[1] + infoADAS13.shape[1]+ infoVentricles.shape[1]-2))
for i in range(infoDX.shape[0]):
	for j in range(6):
		FullDX[i*6+j,:infoDX.shape[1]] = infoDX[i,:]
		FullDX[i*6+j,infoDX.shape[1]:infoDX.shape[1] + infoADAS13.shape[1]-1] = infoADAS13[i,1:]
		FullDX[i*6+j,infoDX.shape[1] + infoADAS13.shape[1]-1:] = infoVentricles[i,1:]


FullDX_withDates  = np.zeros((FullDX.shape[0],FullDX.shape[1]+1),dtype=info.dtype)

FullDX_withDates[:,0:2] = ListOFMonth
FullDX_withDates[:,2:] =  FullDX[:,1:]



RID_LB = LeaderBoradTemplateDF.columns.get_loc("RID")
EXAMDATE_LB =  LeaderBoradTemplateDF.columns.get_loc("Forecast Date")
CN_LB = LeaderBoradTemplateDF.columns.get_loc("CN relative probability")


MCI_LB=LeaderBoradTemplateDF.columns.get_loc("MCI relative probability")
AD_LB = LeaderBoradTemplateDF.columns.get_loc("AD relative probability")
ADAS13_LB = LeaderBoradTemplateDF.columns.get_loc("ADAS13")
ADAS13_lower50_LB = LeaderBoradTemplateDF.columns.get_loc("ADAS13 50% CI lower")
ADAS13_upper50_LB =LeaderBoradTemplateDF.columns.get_loc("ADAS13 50% CI upper")	
Ventricles_LB = LeaderBoradTemplateDF.columns.get_loc("Ventricles_ICV")
Ventricles_lower50_LB  =LeaderBoradTemplateDF.columns.get_loc("Ventricles_ICV 50% CI lower")
Ventricles_upper50_LB  =LeaderBoradTemplateDF.columns.get_loc("Ventricles_ICV 50% CI upper")


infoLB=LeaderBoradTemplateDF.values

start=0
for i in range(infoLB.shape[0]): 
	for j in range(start , FullDX_withDates.shape[0]):
		if(FullDX_withDates[j,0]==infoLB[i,RID_LB] and to_datetime(FullDX_withDates[j,1]) == to_datetime(infoLB[i,EXAMDATE_LB])):
			infoLB[i,CN_LB:] = FullDX_withDates[j,2:]
		elif(FullDX_withDates[j,0]==infoLB[i,RID_LB] and to_datetime(FullDX_withDates[j,1]) >to_datetime(infoLB[i,EXAMDATE_LB])):
			start=j
			break
		elif(FullDX_withDates[j,0]>infoLB[i,RID_LB] ):
			start=j
			break

RIDDXLB = dataframeDXLB.columns.get_loc("RID")
VISCODEDXLB = dataframeDXLB.columns.get_loc("VISCODE")
DX_indxLB= dataframeDXLB.columns.get_loc("DX")


RIDADAS13LB = dataframeADAS13LB.columns.get_loc("RID")
VISCODEADAS13LB = dataframeADAS13LB.columns.get_loc("VISCODE")
ADAS13_indxLB = dataframeADAS13LB.columns.get_loc("ADAS13")


RIDVentriclesLB = dataframeVentriclesLB.columns.get_loc("RID")
VISCODEVentriclesLB = dataframeVentriclesLB.columns.get_loc("VISCODE")
Ventricles_indxLB =  dataframeVentriclesLB.columns.get_loc("Ventricles")

plusADAS13 = 1.57015157
minusADAS13 = 0.03725114
plusVentricles = 0.002510001
minusVentricles = 0.000818025



OldForecastValues =  np.zeros((info.shape[0],11) ,dtype=info.dtype)


OldForecastValues[:,0]=info[:,0]
for i in range (info.shape[0]):
	OldForecastValues[i,1]=datetime(info[i,EXAMDATE].year ,  info[i,EXAMDATE].month,1)


for i in range (info.shape[0]):
	for j in range(infoDXLB.shape[0]):
		if(info[i,RID]==infoDXLB[j,RIDDXLB] and info[i,VISCODE]==infoDXLB[j,VISCODEDXLB]):
			if(infoDXLB[j,DX_indxLB]==1):
				OldForecastValues[i,2] = 1
				OldForecastValues[i,3]=0
				OldForecastValues[i,4] = 0
			elif(infoDXLB[j,DX_indxLB]==2):
				OldForecastValues[i,2] = 0
				OldForecastValues[i,3]=1
				OldForecastValues[i,4] = 0
			elif(infoDXLB[j,DX_indxLB]==3):
				OldForecastValues[i,2] = 0
				OldForecastValues[i,3]=0
				OldForecastValues[i,4] = 1


for i in range (info.shape[0]):
	for j in range(infoADAS13LB.shape[0]):
		if(info[i,RID]==infoADAS13LB[j,RIDADAS13LB] and info[i,VISCODE]==infoADAS13LB[j,VISCODEADAS13LB]):
			OldForecastValues[i,5] = infoADAS13LB[j,ADAS13_indxLB]
			OldForecastValues[i,6] = infoADAS13LB[j,ADAS13_indxLB]-minusADAS13
			OldForecastValues[i,7] = infoADAS13LB[j,ADAS13_indxLB]+plusADAS13


for i in range (info.shape[0]):
	for j in range(infoVentriclesLB.shape[0]):
		if(info[i,RID]==infoVentriclesLB[j,RIDVentriclesLB] and info[i,VISCODE]==infoVentriclesLB[j,VISCODEVentriclesLB]):
			OldForecastValues[i,8] = infoVentriclesLB[j,Ventricles_indxLB]
			OldForecastValues[i,9] = infoVentriclesLB[j,Ventricles_indxLB]-minusVentricles
			OldForecastValues[i,10] = infoVentriclesLB[j,Ventricles_indxLB]+plusVentricles



col=[]
for i in range(LeaderBoradTemplateDF.columns.shape[0]):
	if LeaderBoradTemplateDF.columns.values[i]!='Forecast Month':
		
		col.append(LeaderBoradTemplateDF.columns.values[i])

df=DataFrame(OldForecastValues, columns=col)
df.sort_values(['RID', 'Forecast Date'], ascending=[True, True], inplace=True)
OldForecastValues=df.values

ExpandedOldForecastValues =  np.ones((OldForecastValues.shape[0]*10,OldForecastValues.shape[1]) ,dtype=info.dtype)*-999999
start_date = to_datetime('20100501', format='%Y%m%d')


RID=0
Date=1
indx = 0
for i in range(OldForecastValues.shape[0]-1):
	RID_current  = OldForecastValues[i,RID]
	date_current =to_datetime(OldForecastValues[i,Date])
	if(OldForecastValues[i+1,RID]==RID_current ):
		while (to_datetime(date_current)<to_datetime(OldForecastValues[i+1,Date])):
			ExpandedOldForecastValues[indx,:] = OldForecastValues[i,:]
			ExpandedOldForecastValues[indx,Date] = date_current
			date_current += relativedelta(months=+1)
			indx+=1
	else:
		for j in range(6):
			ExpandedOldForecastValues[indx,:] = OldForecastValues[i,:]
			ExpandedOldForecastValues[indx,Date] = date_current
			date_current += relativedelta(months=+1)
			indx+=1
	print (date_current , start_date, OldForecastValues[i,RID] , OldForecastValues[i+1,RID]) 
	if(OldForecastValues[i+1,RID]!=OldForecastValues[i,RID] or i==(OldForecastValues.shape[0]-2)):
		while (to_datetime(date_current)<=to_datetime(start_date)):
			ExpandedOldForecastValues[indx,:] = OldForecastValues[i,:]
			ExpandedOldForecastValues[indx,Date] = date_current
			date_current += relativedelta(months=+1)
			indx+=1







df=DataFrame(ExpandedOldForecastValues)
df.replace('-999999', np.nan, inplace=True)
ExpandedOldForecastValues = df.values



count=0
start=0
for i in range(infoLB.shape[0]): 
	for j in range(start , ExpandedOldForecastValues.shape[0]):
		
		if(ExpandedOldForecastValues[j,0]==infoLB[i,RID_LB] and to_datetime(ExpandedOldForecastValues[j,1]) == to_datetime(infoLB[i,EXAMDATE_LB])):
			if(np.isnan(infoLB[i,CN_LB]) and np.isnan(infoLB[i,MCI_LB]) and np.isnan(infoLB[i,AD_LB]) and 
				np.isnan(infoLB[i,ADAS13_LB]) and np.isnan(infoLB[i,ADAS13_lower50_LB]) and np.isnan(infoLB[i,ADAS13_upper50_LB]) and
				np.isnan(infoLB[i,Ventricles_LB]) and np.isnan(infoLB[i,Ventricles_lower50_LB]) and np.isnan(infoLB[i,Ventricles_upper50_LB])):
					count+=1
					infoLB[i,CN_LB:] = ExpandedOldForecastValues[j,2:]
		elif(ExpandedOldForecastValues[j,0]==infoLB[i,RID_LB] and to_datetime(ExpandedOldForecastValues[j,1]) >to_datetime(infoLB[i,EXAMDATE_LB])):
			start=j
			break
		elif(ExpandedOldForecastValues[j,0]>infoLB[i,RID_LB] ):
			start=j
			break





for i in range(1,infoLB.shape[0]):
	if(np.isnan(infoLB[i,CN_LB]) and np.isnan(infoLB[i,MCI_LB]) and np.isnan(infoLB[i,AD_LB]) and 
				np.isnan(infoLB[i,ADAS13_LB]) and np.isnan(infoLB[i,ADAS13_lower50_LB]) and np.isnan(infoLB[i,ADAS13_upper50_LB]) and
				np.isnan(infoLB[i,Ventricles_LB]) and np.isnan(infoLB[i,Ventricles_lower50_LB]) and np.isnan(infoLB[i,Ventricles_upper50_LB])):
		if(infoLB[i,RID_LB]==infoLB[i-1,RID_LB]):
			infoLB[i,CN_LB:] = infoLB[i-1,CN_LB:]



df=DataFrame(infoLB, columns=LeaderBoradTemplateDF.columns.values)
df.to_csv('/Users/aya/Documents/Research/Alzheimer/tadpole_challenge/ForecastProcessed_data/TADPOLE_Submission_Leaderboard_BravoLab.csv',index=False)


