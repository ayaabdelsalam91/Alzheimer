import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame
import numpy as np
import getVentriclesAndADASData as getData
import getTimeSeries as TS
from getLSTMTrainingandTesting  import  getLSTMInput , FillinDataForForecasting ,DuplicateDataForForcasting  ,getLSTMinXTimeSteps
from sklearn.model_selection import train_test_split




features = ["RAVLT_immediate_bl" ,  
"RAVLT_immediate",
"DXCHANGE",
"MMSE",
"MMSE_bl",
"CDRSB_bl"]




# getData.ChangeFeatures(features, "FeaturesToCategories" ,  "TADPOLETraining" ,"ToForecastLeaderBoardTrainModifiedFeatures")
# getData.ChangeFeatures(features, "FeaturesToCategories" ,  "TADPOLETesting" ,"ToForecastLeaderBoardModifiedFeatures")
# getData.ChangeFeatures(features, "FeaturesToCategories" ,  "TADPOLED3" ,"ToForecastLeaderBoardModifiedFeaturesD3")


getData.getDataOfIntrest("ADAS13" ,"TADPOLED3" , "ADAS13FeaturesForecastLeaderBoardD3", otherFeatures = features )# ,detailFile ="FeaturesToCategories")
getData.getDataOfIntrest("ADAS13" ,"TADPOLETesting" , "ADAS13FeaturesForecastLeaderBoard", otherFeatures = features) # ,detailFile ="FeaturesToCategories")
getData.getDataOfIntrest("ADAS13" ,"TADPOLETraining" , "ADAS13FeaturesForecastLeaderBoardTrain", otherFeatures = features) # ,detailFile ="FeaturesToCategories")



TS.getTimeSeries("ADAS13","ADAS13FeaturesForecastLeaderBoardTrain" , "ADAS13FeaturesForecastLeaderBoardTrainTS" ,multivariate=True) 
TS.getTimeSeries("ADAS13","ADAS13FeaturesForecastLeaderBoard" , "ADAS13FeaturesForecastLeaderBoardTS" ,multivariate=True) 
TS.getTimeSeries("ADAS13","ADAS13FeaturesForecastLeaderBoardD3" , "ADAS13FeaturesForecastLeaderBoardTSD3" ,multivariate=True) 


FillinDataForForecasting("ADAS13","ADAS13FeaturesForecastLeaderBoardTrainTS","LSTMDataADAS13ForecastLeaderBoardTrain",multivariate=True)
FillinDataForForecasting("ADAS13","ADAS13FeaturesForecastLeaderBoardTS","LSTMDataADAS13ForecastLeaderBoard",multivariate=True,Testing = True)
FillinDataForForecasting("ADAS13","ADAS13FeaturesForecastLeaderBoardTSD3","LSTMDataADAS13ForecastLeaderBoardD3",multivariate=True,Testing = True)

getLSTMInput(  "LSTMDataADAS13ForecastLeaderBoardTrain","LSTMDataADAS13ForecastLeaderBoardTrainParsed" ,DiffTarget=False)
getLSTMInput( "LSTMDataADAS13ForecastLeaderBoard","LSTMDataADAS13ForecastLeaderBoardParsed" , Testing=True )
getLSTMInput( "LSTMDataADAS13ForecastLeaderBoardD3","LSTMDataADAS13ForecastLeaderBoardParsedD3" , Testing=True )
