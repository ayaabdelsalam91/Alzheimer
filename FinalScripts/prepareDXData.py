
import getVentriclesAndADASData as getData
import getTimeSeries as TS
from getLSTMTrainingandTesting  import  getLSTMInput , FillinDataForForecasting ,DuplicateDataForForcasting  ,getLSTMinXTimeSteps
from pandas import read_csv, DataFrame



print ("HERe!!!")
features = ["ADAS13" , "ADAS13_bl" , "ADAS11" ,"ADAS11_bl"]
getData.getDataOfIntrest("CDRSB","TADPOLED3" , "CDRSBFeaturesForecastLeaderBoardD3", otherFeatures = features) 
getData.getDataOfIntrest("CDRSB","TADPOLETesting" , "CDRSBFeaturesForecastLeaderBoard", otherFeatures = features) 
getData.getDataOfIntrest("CDRSB","TADPOLETraining" , "CDRSBFeaturesForecastLeaderBoardTrain", otherFeatures = features) 


data ='../Finaldata/CDRSBFeaturesForecastLeaderBoardD3.csv'
dataframe = read_csv(data, names=None)
print("CDRSBFeaturesForecastLeaderBoardD3" , dataframe.shape)
data ='../Finaldata/CDRSBFeaturesForecastLeaderBoard.csv'
dataframe = read_csv(data, names=None)
print("CDRSBFeaturesForecastLeaderBoard" , dataframe.shape)
data ='../Finaldata/CDRSBFeaturesForecastLeaderBoardTrain.csv'
dataframe = read_csv(data, names=None)
print("CDRSBFeaturesForecastLeaderBoardTrain" , dataframe.shape)


getData.getDataOfIntrest("MMSE","TADPOLED3" , "MMSEFeaturesForecastLeaderBoardD3") 
getData.getDataOfIntrest("MMSE","TADPOLETesting" , "MMSEFeaturesForecastLeaderBoard") 
getData.getDataOfIntrest("MMSE","TADPOLETraining" , "MMSEFeaturesForecastLeaderBoardTrain") 


data ='../Finaldata/MMSEFeaturesForecastLeaderBoardD3.csv'
dataframe = read_csv(data, names=None)
print("MMSEFeaturesForecastLeaderBoardD3" , dataframe.shape)
data ='../Finaldata/MMSEFeaturesForecastLeaderBoard.csv'
dataframe = read_csv(data, names=None)
print("MMSEFeaturesForecastLeaderBoard" , dataframe.shape)
data ='../Finaldata/MMSEFeaturesForecastLeaderBoardTrain.csv'
dataframe = read_csv(data, names=None)
print("MMSEFeaturesForecastLeaderBoardTrain" , dataframe.shape)


features = ["CDRSB_bl"]
getData.getDataOfIntrest("DX","TADPOLED3" , "DXFeaturesForecastLeaderBoardD3", otherFeatures = features)
getData.getDataOfIntrest("DX","TADPOLETesting" , "DXFeaturesForecastLeaderBoard", otherFeatures = features) 
getData.getDataOfIntrest("DX","TADPOLETraining" , "DXFeaturesForecastLeaderBoardTrain", otherFeatures = features) 


data ='../Finaldata/DXFeaturesForecastLeaderBoardD3.csv'
dataframe = read_csv(data, names=None)
print("DXFeaturesForecastLeaderBoardD3" , dataframe.shape)
data ='../Finaldata/DXFeaturesForecastLeaderBoard.csv'
dataframe = read_csv(data, names=None)
print("DXFeaturesForecastLeaderBoard" , dataframe.shape)
data ='../Finaldata/DXFeaturesForecastLeaderBoardTrain.csv'
dataframe = read_csv(data, names=None)
print("DXFeaturesForecastLeaderBoardTrain" , dataframe.shape)




TS.getTimeSeries("DX","DXFeaturesForecastLeaderBoardD3" , "DXFeaturesForecastLeaderBoardTSD3" , multivariate=True) 
TS.getTimeSeries("DX","DXFeaturesForecastLeaderBoard" , "DXFeaturesForecastLeaderBoardTS" , multivariate=True) 
TS.getTimeSeries("DX","DXFeaturesForecastLeaderBoardTrain" , "DXFeaturesForecastLeaderBoardTrainTS" , multivariate=True)

data ='../Finaldata/DXFeaturesForecastLeaderBoardTSD3.csv'
dataframe = read_csv(data, names=None)
print("DXFeaturesForecastLeaderBoardTSD3" , dataframe.shape)
data ='../Finaldata/DXFeaturesForecastLeaderBoardTS.csv'
dataframe = read_csv(data, names=None)
print("DXFeaturesForecastLeaderBoardTS" , dataframe.shape)
data ='../Finaldata/DXFeaturesForecastLeaderBoardTrainTS.csv'
dataframe = read_csv(data, names=None)
print("DXFeaturesForecastLeaderBoardTrainTS" , dataframe.shape)





TS.getTimeSeries("MMSE","MMSEFeaturesForecastLeaderBoardD3" , "MMSEFeaturesForecastLeaderBoardTSD3" ) 
TS.getTimeSeries("MMSE","MMSEFeaturesForecastLeaderBoard" , "MMSEFeaturesForecastLeaderBoardTS" ) 
TS.getTimeSeries("MMSE","MMSEFeaturesForecastLeaderBoardTrain" , "MMSEFeaturesForecastLeaderBoardTrainTS")


data ='../Finaldata/MMSEFeaturesForecastLeaderBoardTSD3.csv'
dataframe = read_csv(data, names=None)
print("MMSEFeaturesForecastLeaderBoardTSD3" , dataframe.shape)
data ='../Finaldata/MMSEFeaturesForecastLeaderBoardTS.csv'
dataframe = read_csv(data, names=None)
print("MMSEFeaturesForecastLeaderBoardTS" , dataframe.shape)
data ='../Finaldata/MMSEFeaturesForecastLeaderBoardTrainTS.csv'
dataframe = read_csv(data, names=None)
print("MMSEFeaturesForecastLeaderBoardTrainTS" , dataframe.shape)




TS.getTimeSeries("CDRSB","CDRSBFeaturesForecastLeaderBoardD3" , "CDRSBFeaturesForecastLeaderBoardTSD3"  ,multivariate=True) 
TS.getTimeSeries("CDRSB","CDRSBFeaturesForecastLeaderBoard" , "CDRSBFeaturesForecastLeaderBoardTS"  ,multivariate=True) 
TS.getTimeSeries("CDRSB","CDRSBFeaturesForecastLeaderBoardTrain" , "CDRSBFeaturesForecastLeaderBoardTrainTS"  ,multivariate=True) 



data ='../Finaldata/CDRSBFeaturesForecastLeaderBoardTSD3.csv'
dataframe = read_csv(data, names=None)
print("CDRSBFeaturesForecastLeaderBoardTSD3" , dataframe.shape)
data ='../Finaldata/CDRSBFeaturesForecastLeaderBoardTS.csv'
dataframe = read_csv(data, names=None)
print("CDRSBFeaturesForecastLeaderBoardTS" , dataframe.shape)
data ='../Finaldata/CDRSBFeaturesForecastLeaderBoardTrainTS.csv'
dataframe = read_csv(data, names=None)
print("CDRSBFeaturesForecastLeaderBoardTrainTS" , dataframe.shape)



FillinDataForForecasting("DX","DXFeaturesForecastLeaderBoardTSD3","LSTMDataDXForecastLeaderBoardD3",multivariate=True,Testing = True)
FillinDataForForecasting("DX","DXFeaturesForecastLeaderBoardTS","LSTMDataDXForecastLeaderBoard",multivariate=True,Testing = True)
FillinDataForForecasting("DX","DXFeaturesForecastLeaderBoardTrainTS","LSTMDataDXForecastLeaderBoardTrain",multivariate=True)


data ='../Finaldata/LSTMDataDXForecastLeaderBoardD3.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataDXForecastLeaderBoardD3" , dataframe.shape)
data ='../Finaldata/LSTMDataDXForecastLeaderBoard.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataDXForecastLeaderBoard" , dataframe.shape)
data ='../Finaldata/LSTMDataDXForecastLeaderBoardTrain.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataDXForecastLeaderBoardTrain" , dataframe.shape)

FillinDataForForecasting("MMSE","MMSEFeaturesForecastLeaderBoardTSD3","LSTMDataMMSEForecastLeaderBoardD3",shift=True,Testing = True)
FillinDataForForecasting("MMSE","MMSEFeaturesForecastLeaderBoardTS","LSTMDataMMSEForecastLeaderBoard",shift=True,Testing = True)
FillinDataForForecasting("MMSE","MMSEFeaturesForecastLeaderBoardTrainTS","LSTMDataMMSEForecastLeaderBoardTrain",shift=True)


data ='../Finaldata/LSTMDataMMSEForecastLeaderBoardD3.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataMMSEForecastLeaderBoardD3" , dataframe.shape)
data ='../Finaldata/LSTMDataMMSEForecastLeaderBoard.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataMMSEForecastLeaderBoard" , dataframe.shape)
data ='../Finaldata/LSTMDataMMSEForecastLeaderBoardTrain.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataMMSEForecastLeaderBoardTrain" , dataframe.shape)

FillinDataForForecasting("CDRSB","CDRSBFeaturesForecastLeaderBoardTSD3","LSTMDataCDRSBForecastLeaderBoardD3",multivariate=True,shift=True,Testing = True)
FillinDataForForecasting("CDRSB","CDRSBFeaturesForecastLeaderBoardTS","LSTMDataCDRSBForecastLeaderBoard",multivariate=True,shift=True,Testing = True)
FillinDataForForecasting("CDRSB","CDRSBFeaturesForecastLeaderBoardTrainTS","LSTMDataCDRSBForecastLeaderBoardTrain",multivariate=True,shift=True)

data ='../Finaldata/LSTMDataCDRSBForecastLeaderBoardD3.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataCDRSBForecastLeaderBoardD3" , dataframe.shape)
data ='../Finaldata/LSTMDataCDRSBForecastLeaderBoard.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataCDRSBForecastLeaderBoard" , dataframe.shape)
data ='../Finaldata/LSTMDataCDRSBForecastLeaderBoardTrain.csv'
dataframe = read_csv(data, names=None)
print("LSTMDataCDRSBForecastLeaderBoardTrain" , dataframe.shape)


getLSTMInput( "LSTMDataDXForecastLeaderBoardD3","LSTMDataDXForecastLeaderBoardParsedD3" , Testing=True )
getLSTMInput( "LSTMDataDXForecastLeaderBoard","LSTMDataDXForecastLeaderBoardParsed" , Testing=True )
getLSTMInput(  "LSTMDataDXForecastLeaderBoardTrain","LSTMDataDXForecastLeaderBoardTrainParsed" ,DiffTarget=False)

getLSTMInput( "LSTMDataMMSEForecastLeaderBoardD3","LSTMDataMMSEForecastLeaderBoardParsedD3" , Testing=True ,shift=True) 
getLSTMInput( "LSTMDataMMSEForecastLeaderBoard","LSTMDataMMSEForecastLeaderBoardParsed" , Testing=True ,shift=True) 
getLSTMInput(  "LSTMDataMMSEForecastLeaderBoardTrain","LSTMDataMMSEForecastLeaderBoardTrainParsed" ,DiffTarget=False ,shift=True)

getLSTMInput( "LSTMDataCDRSBForecastLeaderBoardD3","LSTMDataCDRSBForecastLeaderBoardParsedD3" , Testing=True ,shift=True) 
getLSTMInput( "LSTMDataCDRSBForecastLeaderBoard","LSTMDataCDRSBForecastLeaderBoardParsed" , Testing=True ,shift=True) 
getLSTMInput(  "LSTMDataCDRSBForecastLeaderBoardTrain","LSTMDataCDRSBForecastLeaderBoardTrainParsed" ,DiffTarget=False,shift=True)




