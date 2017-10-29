import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame
import numpy as np
import getVentriclesAndADASData as getData
import getTimeSeries as TS
from getLSTMTrainingandTesting  import  getLSTMInput , FillinDataForForecasting ,DuplicateDataForForcasting ,  getTheLastXTimeStepsForForecasting
from sklearn.model_selection import train_test_split

# features = ["RAVLT_immediate_bl" , 
# "RAVLT_immediate",
# "DXCHANGE",
# "MMSE",
# "MMSE_bl",
# "CDRSB_bl"]
# getData.getDataOfIntrest("ADAS13" ,"ToForecastLeaderBoard" , "ADAS13FeaturesForecastLeaderBoard", features)
# getData.getDataOfIntrest("ADAS13" ,"ToForecastLeaderBoardTrain" , "ADAS13FeaturesForecastLeaderBoardTrain", features)
# features = ["ADAS13" , "ADAS13_bl" , "ADAS11" ,"ADAS11_bl" ,"MMSE"]
# getData.getDataOfIntrest("CDRSB","ToForecastLeaderBoard" , "CDRSBFeaturesForecastLeaderBoard",features) 
# getData.getDataOfIntrest("CDRSB","ToForecastLeaderBoardTrain" , "CDRSBFeaturesForecastLeaderBoardTrain",features) 


# features = ["ADAS13" , "ADAS13_bl" , "ADAS11" ,"ADAS11_bl" ,"CDRSB" , "CDRSB_bl"]
# getData.getDataOfIntrest("MMSE","ToForecastLeaderBoard" , "MMSEFeaturesForecastLeaderBoard",features) 
# getData.getDataOfIntrest("MMSE","ToForecastLeaderBoardTrain" , "MMSEFeaturesForecastLeaderBoardTrain",features) 



# features =["ST37SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
# "ST96SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
# "Ventricles_bl"]
# getData.getDataOfIntrest("Ventricles","ToForecastLeaderBoard" , "VentriclesFeaturesForecastLeaderBoard",features) 
# getData.getDataOfIntrest("Ventricles","ToForecastLeaderBoardTrain" , "VentriclesFeaturesForecastLeaderBoardTrain",features) 

# features = ["CDRSB_bl"]
# getData.getDataOfIntrest("DX","ToForecastLeaderBoard" , "DXFeaturesForecastLeaderBoard",features) 
# getData.getDataOfIntrest("DX","ToForecastLeaderBoardTrain" , "DXFeaturesForecastLeaderBoardTrain",features) 

# TS.getTimeSeries("DX","DXFeaturesForecastLeaderBoardTrain" , "DXFeaturesForecastLeaderBoardTrainTS" , multivariate=True) 
# TS.getTimeSeries("Ventricles","VentriclesFeaturesForecastLeaderBoardTrain" , "VentriclesFeaturesForecastLeaderBoardTrainTS" ,multivariate=True) 
# TS.getTimeSeries("ADAS13","ADAS13FeaturesForecastLeaderBoardTrain" , "ADAS13FeaturesForecastLeaderBoardTrainTS" ,multivariate=True) 
# TS.getTimeSeries("MMSE","MMSEFeaturesForecastLeaderBoardTrain" , "MMSEFeaturesForecastLeaderBoardTrainTS" ,multivariate=True) 
# TS.getTimeSeries("CDRSB","CDRSBFeaturesForecastLeaderBoardTrain" , "CDRSBFeaturesForecastLeaderBoardTrainTS"  ,multivariate=True) 

# TS.getTimeSeries("DX","DXFeaturesForecastLeaderBoard" , "DXFeaturesForecastLeaderBoardTS" , multivariate=True) 
# TS.getTimeSeries("Ventricles","VentriclesFeaturesForecastLeaderBoard" , "VentriclesFeaturesForecastLeaderBoardTS" ,multivariate=True) 
# TS.getTimeSeries("ADAS13","ADAS13FeaturesForecastLeaderBoard" , "ADAS13FeaturesForecastLeaderBoardTS" ,multivariate=True) 
# TS.getTimeSeries("MMSE","MMSEFeaturesForecastLeaderBoard" , "MMSEFeaturesForecastLeaderBoardTS" ,multivariate=True) 
# TS.getTimeSeries("CDRSB","CDRSBFeaturesForecastLeaderBoard" , "CDRSBFeaturesForecastLeaderBoardTS"  ,multivariate=True) 


#########################


# FillinDataForForecasting("Ventricles","VentriclesFeaturesForecastLeaderBoardTrainTS","LSTMDataVentriclesForecastLeaderBoardTrain",multivariate=True)
# FillinDataForForecasting("Ventricles","VentriclesFeaturesForecastLeaderBoardTS","LSTMDataVentriclesForecastLeaderBoard",multivariate=True,Testing = True)
# FillinDataForForecasting("DX","DXFeaturesForecastLeaderBoardTrainTS","LSTMDataDXForecastLeaderBoardTrain",multivariate=True)
# FillinDataForForecasting("DX","DXFeaturesForecastLeaderBoardTS","LSTMDataDXForecastLeaderBoard",multivariate=True,Testing = True)
# FillinDataForForecasting("ADAS13","ADAS13FeaturesForecastLeaderBoardTrainTS","LSTMDataADAS13ForecastLeaderBoardTrain",multivariate=True)
# FillinDataForForecasting("ADAS13","ADAS13FeaturesForecastLeaderBoardTS","LSTMDataADAS13ForecastLeaderBoard",multivariate=True,Testing = True)
# FillinDataForForecasting("MMSE","MMSEFeaturesForecastLeaderBoardTrainTS","LSTMDataMMSEForecastLeaderBoardTrain",multivariate=True,shift=True) 
# FillinDataForForecasting("MMSE","MMSEFeaturesForecastLeaderBoardTS","LSTMDataMMSEForecastLeaderBoard",multivariate=True,shift=True,Testing = True)
# FillinDataForForecasting("CDRSB","CDRSBFeaturesForecastLeaderBoardTrainTS","LSTMDataCDRSBForecastLeaderBoardTrain",multivariate=True,shift=True) 
# FillinDataForForecasting("CDRSB","CDRSBFeaturesForecastLeaderBoardTS","LSTMDataCDRSBForecastLeaderBoard",multivariate=True,shift=True,Testing = True)


getLSTMInput(  "LSTMDataVentriclesForecastLeaderBoardTrain","LSTMDataVentriclesForecastLeaderBoardTrainParsed" ,DiffTarget=True)
getLSTMInput( "LSTMDataVentriclesForecastLeaderBoard","LSTMDataVentriclesForecastLeaderBoardParsed" , Testing=True )

getLSTMInput(  "LSTMDataDXForecastLeaderBoardTrain","LSTMDataDXForecastLeaderBoardTrainParsed" ,DiffTarget=False)
getLSTMInput( "LSTMDataDXForecastLeaderBoard","LSTMDataDXForecastLeaderBoardParsed" , Testing=True )

getLSTMInput(  "LSTMDataADAS13ForecastLeaderBoardTrain","LSTMDataADAS13ForecastLeaderBoardTrainParsed" ,DiffTarget=True)
getLSTMInput( "LSTMDataADAS13ForecastLeaderBoard","LSTMDataADAS13ForecastLeaderBoardParsed" , Testing=True )

getLSTMInput(  "LSTMDataMMSEForecastLeaderBoardTrain","LSTMDataMMSEForecastLeaderBoardTrainParsed" ,DiffTarget=False ,shift=True) 
getLSTMInput( "LSTMDataMMSEForecastLeaderBoard","LSTMDataMMSEForecastLeaderBoardParsed" , Testing=True ,shift=True) 

getLSTMInput(  "LSTMDataCDRSBForecastLeaderBoardTrain","LSTMDataCDRSBForecastLeaderBoardTrainParsed" ,DiffTarget=False,shift=True) 
getLSTMInput( "LSTMDataCDRSBForecastLeaderBoard","LSTMDataCDRSBForecastLeaderBoardParsed" , Testing=True ,shift=True) 



# DuplicateDataForForcasting("LSTMDataVentriclesForecastLeaderBoardTrain","LSTMDataVentriclesForecastLeaderBoardTrainParsed" , "Ventricles" , multivariate=True)
# DuplicateDataForForcasting("LSTMDataVentriclesForecastLeaderBoard","LSTMDataVentriclesForecastLeaderBoardParsed" , "Ventricles" , multivariate=True)

# DuplicateData("Training","DX", multivariate=True , isnotDx=False)
# DuplicateData("Training","Ventricles" ,  multivariate=True)
# DuplicateData("Training","ADAS13" ,  multivariate=True)
# DuplicateData("Training","MMSE" ,multivariate=True,shift=True,isnotDx=False)
# DuplicateData("Training","CDRSB",multivariate=True,shift=True,isnotDx=False)

