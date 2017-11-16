#import deleteUneededcol_beforeSplit
# from replaceTextWithNumber import replaceTextWithNumber
# replaceTextWithNumber("TADPOLE_D1_D2_beforeColDel" , "TADPOLE_processed")


from preprocessing import Preprocessing
import ImputationMethods as IM
from pandas import read_csv
from  splitFiles import split

# Preprocessing("TADPOLE_processed" , "preprocessedTADPOLE")



# data = '../Finaldata/preprocessedTADPOLE.csv'
# dataframe = read_csv(data, names=None)
# df = IM.HotDeckImputation(dataframe)
# df.to_csv('../Finaldata/TADPOLEFinal.csv',index=False)


# split("TADPOLEFinal" , "TADPOLETraining","TADPOLETesting" , "TADPOLED3")


# import prepareVentriclesData
# import prepareADAS13Data
import prepareDXData

# import VentriclesLSTM
# import ADAS13LSTM
# import DXLSTM


# replaceTextWithNumber("ForecastD2" , "ForecastD2")
# replaceTextWithNumber("ForecastD3" , "ForecastD3")



# from generateForecast import generateForecast
# generateForecast("ForecastD2",
#  	"DXOutputPersistence","ADAS13OutputPersistence","VentriclesOutputPersistence",
#  	"DXFeaturesForecastLeaderBoard","ADAS13FeaturesForecastLeaderBoard","VentriclesFeaturesForecastLeaderBoard", 
#  	1.296486616,0.09537667 ,0.000875521 , 0.000320904 ,
#  	"TADPOLE_Submission_BravoLab")



# generateForecast("ForecastD3",
#  	"DXD3Persistence","ADAS13D3Persistence","VentriclesD3Persistence",
#  	"DXFeaturesForecastLeaderBoardD3","ADAS13FeaturesForecastLeaderBoardD3","VentriclesFeaturesForecastLeaderBoardD3", 
#  	1.296486616,0.09537667 ,0.000875521 , 0.000320904 ,
#  	"TADPOLE_Submission_BravoLab_D3")
