import numpy as np
from LSTM_ADAS13BEST  import LSTM
from getLSTMTrainingandTesting  import getValueOfIntrest


List = [ "ADAS13FeaturesForecastLeaderBoardTrain"]
ValuesList, BaseLineList   =  getValueOfIntrest(List)# , detailFile = "FeaturesToCategories" , Orignal = "ToForecastLeaderBoardTrainModifiedFeatures")
print(ValuesList , BaseLineList)
LSTM("ADAS13" , "LSTMDataADAS13ForecastLeaderBoardTrainParsed" , "LSTMDataADAS13ForecastLeaderBoardParsed" ,"LSTMDataADAS13ForecastLeaderBoardParsedD3" ,  Validation=False ,  DefaultValues = ValuesList , isBaseline = BaseLineList )
