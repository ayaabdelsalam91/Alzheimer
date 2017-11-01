import numpy as np
from LSTM_Models import LSTM
from LSTMDX4 import LSTMDX
from LSTM_ModelsOld import LSTM as LSTM2


#LSTM2("ADAS13", "LSTMDataADAS13ForecastLeaderBoardTrainParsed" , "LSTMDataADAS13ForecastLeaderBoardParsed" , Validation=True)
#LSTM("Ventricles","LSTMDataVentriclesForecastLeaderBoardTrainParsed" , "LSTMDataVentriclesForecastLeaderBoardParsed",Validation=True )
LSTMDX("LSTMDataDXForecastLeaderBoardTrainParsed" , "LSTMDataCDRSBForecastLeaderBoardTrainParsed" , "LSTMDataMMSEForecastLeaderBoardTrainParsed",
  "LSTMDataDXForecastLeaderBoardParsed" , "LSTMDataCDRSBForecastLeaderBoardParsed" , "LSTMDataMMSEForecastLeaderBoardParsed", Validation=False)