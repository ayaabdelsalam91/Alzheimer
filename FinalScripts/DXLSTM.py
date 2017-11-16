from LSTM_DXBEST import LSTMDX
from getLSTMTrainingandTesting  import getValueOfIntrest






List = [ "DXFeaturesForecastLeaderBoardTrain","CDRSBFeaturesForecastLeaderBoardTrain"  , "MMSEFeaturesForecastLeaderBoardTrain"]
ValuesList, BaseLineList =  getValueOfIntrest(List )
print (ValuesList , BaseLineList)
LSTMDX("LSTMDataDXForecastLeaderBoardTrainParsed" , "LSTMDataCDRSBForecastLeaderBoardTrainParsed" , "LSTMDataMMSEForecastLeaderBoardTrainParsed",
 "LSTMDataDXForecastLeaderBoardParsed" , "LSTMDataCDRSBForecastLeaderBoardParsed" , "LSTMDataMMSEForecastLeaderBoardParsed", "LSTMDataDXForecastLeaderBoardParsedD3" , "LSTMDataCDRSBForecastLeaderBoardParsedD3" , "LSTMDataMMSEForecastLeaderBoardParsedD3", 
 Validation=False , DefaultValues = ValuesList , isBaseline = BaseLineList)