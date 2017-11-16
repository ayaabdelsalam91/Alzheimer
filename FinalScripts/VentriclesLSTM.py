import numpy as np
from LSTM_VentriclesBEST  import LSTM
from getLSTMTrainingandTesting  import getValueOfIntrest


List = [ "VentriclesFeaturesForecastLeaderBoardTrain"]
ValuesList, BaseLineList =  getValueOfIntrest(List)
print (ValuesList , BaseLineList)
LSTM("Ventricles","LSTMDataVentriclesForecastLeaderBoardTrainParsed" , "LSTMDataVentriclesForecastLeaderBoardParsed" ,"LSTMDataVentriclesForecastLeaderBoardParsedD3",   Validation=False ,  DefaultValues = ValuesList , isBaseline = BaseLineList)
