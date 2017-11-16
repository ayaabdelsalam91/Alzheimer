import getVentriclesAndADASData as getData
import getTimeSeries as TS
from getLSTMTrainingandTesting  import  getLSTMInput , FillinDataForForecasting ,DuplicateDataForForcasting  ,getLSTMinXTimeSteps
from pandas import read_csv, DataFrame

features =["ST37SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
"ST96SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
"ST30SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
"ST89SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
"ST127SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
"Ventricles_bl", 
"ST37SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16" ,
 "ST96SV_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
  "ST7SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16" , 
 "RIGHT_LATERAL_VENTRICLE_SIZE_UCBERKELEYAV45_10_17_16" , 
 "LEFT_LATERAL_VENTRICLE_SIZE_UCBERKELEYAV45_10_17_16"]


# data ='../Finaldata/TADPOLE_D3.csv'
# dataframe = read_csv(data, names=None)
# print("TADPOLED3" , dataframe.shape)

getData.getDataOfIntrest("Ventricles","TADPOLED3" , "VentriclesFeaturesForecastLeaderBoardD3" , otherFeatures =  features) 
getData.getDataOfIntrest("Ventricles","TADPOLETesting" , "VentriclesFeaturesForecastLeaderBoard" , otherFeatures =  features) 
getData.getDataOfIntrest("Ventricles","TADPOLETraining" , "VentriclesFeaturesForecastLeaderBoardTrain" ,  otherFeatures = features) 

# data ='../Finaldata/VentriclesFeaturesForecastLeaderBoardD3.csv'
# dataframe = read_csv(data, names=None)
# print("VentriclesFeaturesForecastLeaderBoardD3" , dataframe.shape)
# data ='../Finaldata/VentriclesFeaturesForecastLeaderBoard.csv'
# dataframe = read_csv(data, names=None)
# print("VentriclesFeaturesForecastLeaderBoard" , dataframe.shape)
# data ='../Finaldata/VentriclesFeaturesForecastLeaderBoardTrain.csv'
# dataframe = read_csv(data, names=None)
#print("VentriclesFeaturesForecastLeaderBoardTrain" , dataframe.shape)


TS.getTimeSeries("Ventricles","VentriclesFeaturesForecastLeaderBoardTrain" , "VentriclesFeaturesForecastLeaderBoardTrainTS", multivariate=True) 
TS.getTimeSeries("Ventricles","VentriclesFeaturesForecastLeaderBoardD3" , "VentriclesFeaturesForecastLeaderBoardTSD3", multivariate=True) 
TS.getTimeSeries("Ventricles","VentriclesFeaturesForecastLeaderBoard" , "VentriclesFeaturesForecastLeaderBoardTS", multivariate=True) 

# data ='../Finaldata/VentriclesFeaturesForecastLeaderBoardTSD3.csv'
# dataframe = read_csv(data, names=None)
# print("VentriclesFeaturesForecastLeaderBoardTSD3" , dataframe.shape)
# data ='../Finaldata/VentriclesFeaturesForecastLeaderBoardTS.csv'
# dataframe = read_csv(data, names=None)
# print("VentriclesFeaturesForecastLeaderBoardTS" , dataframe.shape)
# data ='../Finaldata/VentriclesFeaturesForecastLeaderBoardTrainTS.csv'
# dataframe = read_csv(data, names=None)
# print("VentriclesFeaturesForecastLeaderBoardTrainTS" , dataframe.shape)

FillinDataForForecasting("Ventricles","VentriclesFeaturesForecastLeaderBoardTrainTS","LSTMDataVentriclesForecastLeaderBoardTrain",multivariate=True)
FillinDataForForecasting("Ventricles","VentriclesFeaturesForecastLeaderBoardTSD3","LSTMDataVentriclesForecastLeaderBoardD3",multivariate=True , Testing = True)
FillinDataForForecasting("Ventricles","VentriclesFeaturesForecastLeaderBoardTS","LSTMDataVentriclesForecastLeaderBoard",multivariate=True , Testing = True)

# data ='../Finaldata/LSTMDataVentriclesForecastLeaderBoardD3.csv'
# dataframe = read_csv(data, names=None)
# print("LSTMDataVentriclesForecastLeaderBoardD3" , dataframe.shape)
# data ='../Finaldata/VentriclesFeaturesForecastLeaderBoard.csv'
# dataframe = read_csv(data, names=None)
# print("LSTMDataVentriclesForecastLeaderBoard" , dataframe.shape)
# data ='../Finaldata/LSTMDataVentriclesForecastLeaderBoard.csv'
# dataframe = read_csv(data, names=None)
# print("LSTMDataVentriclesForecastLeaderBoardTrain" , dataframe.shape)



getLSTMInput(  "LSTMDataVentriclesForecastLeaderBoardTrain","LSTMDataVentriclesForecastLeaderBoardTrainParsed" ,DiffTarget=False)
getLSTMInput( "LSTMDataVentriclesForecastLeaderBoardD3","LSTMDataVentriclesForecastLeaderBoardParsedD3" , Testing=True )
getLSTMInput( "LSTMDataVentriclesForecastLeaderBoard","LSTMDataVentriclesForecastLeaderBoardParsed" , Testing=True )


# data ='../Finaldata/LSTMDataVentriclesForecastLeaderBoardParsedD3.csv'
# dataframe = read_csv(data, names=None)
# print("LSTMDataVentriclesForecastLeaderBoardParsedD3" , dataframe.shape)
# data ='../Finaldata/LSTMDataVentriclesForecastLeaderBoardParsed.csv'
# dataframe = read_csv(data, names=None)
# print("LSTMDataVentriclesForecastLeaderBoardParsed" , dataframe.shape)
# data ='../Finaldata/LSTMDataVentriclesForecastLeaderBoardTrainParsed.csv'
# dataframe = read_csv(data, names=None)
# print("LSTMDataVentriclesForecastLeaderBoardTrainParsed" , dataframe.shape)


