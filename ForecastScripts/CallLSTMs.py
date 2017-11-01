import numpy as np
from LSTM_Models import LSTM
from LSTMDX4 import LSTMDX


#LSTM("ADAS13", "LSTMDataADAS13ForecastLeaderBoardTrainParsed" , "LSTMDataADAS13ForecastLeaderBoardParsed" , Validation=True)
LSTM("Ventricles","LSTMDataVentriclesForecastLeaderBoardTrainParsed" , "LSTMDataVentriclesForecastLeaderBoardParsed",Validation=True )
# LSTMDX("LSTMDataDXForecastLeaderBoardTrainParsed" , "LSTMDataCDRSBForecastLeaderBoardTrainParsed" , "LSTMDataMMSEForecastLeaderBoardTrainParsed",
#  "LSTMDataDXForecastLeaderBoardParsed" , "LSTMDataCDRSBForecastLeaderBoardParsed" , "LSTMDataMMSEForecastLeaderBoardParsed")