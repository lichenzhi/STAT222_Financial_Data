from rename_column import *

def createV3():
    data = rename_data()
    ASK_maxmindiff=data.loc[:,"ASK_PRICE10"]-data.loc[:,"ASK_PRICE1"]
    BID_maxmindiff=data.loc[:,"BID_PRICE1"]-data.loc[:,"BID_PRICE10"]
    
    ASK_PRICE_21_diff=np.absolute(data.loc[:,"ASK_PRICE2"]-data.loc[:,"ASK_PRICE1"])
    ASK_PRICE_32_diff=np.absolute(data.loc[:,"ASK_PRICE3"]-data.loc[:,"ASK_PRICE2"])
    ASK_PRICE_43_diff=np.absolute(data.loc[:,"ASK_PRICE4"]-data.loc[:,"ASK_PRICE3"])
    ASK_PRICE_54_diff=np.absolute(data.loc[:,"ASK_PRICE5"]-data.loc[:,"ASK_PRICE4"])
    ASK_PRICE_65_diff=np.absolute(data.loc[:,"ASK_PRICE6"]-data.loc[:,"ASK_PRICE5"])
    ASK_PRICE_76_diff=np.absolute(data.loc[:,"ASK_PRICE7"]-data.loc[:,"ASK_PRICE6"])
    ASK_PRICE_87_diff=np.absolute(data.loc[:,"ASK_PRICE8"]-data.loc[:,"ASK_PRICE7"])
    ASK_PRICE_98_diff=np.absolute(data.loc[:,"ASK_PRICE9"]-data.loc[:,"ASK_PRICE8"])
    ASK_PRICE_109_diff=np.absolute(data.loc[:,"ASK_PRICE10"]-data.loc[:,"ASK_PRICE9"])
    
    BID_PRICE_21_diff=np.absolute(data.loc[:,"BID_PRICE2"]-data.loc[:,"BID_PRICE1"])
    BID_PRICE_32_diff=np.absolute(data.loc[:,"BID_PRICE3"]-data.loc[:,"BID_PRICE2"])
    BID_PRICE_43_diff=np.absolute(data.loc[:,"BID_PRICE4"]-data.loc[:,"BID_PRICE3"])
    BID_PRICE_54_diff=np.absolute(data.loc[:,"BID_PRICE5"]-data.loc[:,"BID_PRICE4"])
    BID_PRICE_65_diff=np.absolute(data.loc[:,"BID_PRICE6"]-data.loc[:,"BID_PRICE5"])
    BID_PRICE_76_diff=np.absolute(data.loc[:,"BID_PRICE7"]-data.loc[:,"BID_PRICE6"])
    BID_PRICE_87_diff=np.absolute(data.loc[:,"BID_PRICE8"]-data.loc[:,"BID_PRICE7"])
    BID_PRICE_98_diff=np.absolute(data.loc[:,"BID_PRICE9"]-data.loc[:,"BID_PRICE8"])
    BID_PRICE_109_diff=np.absolute(data.loc[:,"BID_PRICE10"]-data.loc[:,"BID_PRICE9"])
    
    v3=pd.DataFrame({"ASK_PRICE_maxmindiff":ASK_maxmindiff,"BID_PRICE_maxmindiff":BID_maxmindiff,
          "ASK_PRICE_21_diff":ASK_PRICE_21_diff,"ASK_PRICE_32_diff":ASK_PRICE_32_diff,
          "ASK_PRICE_43_diff":ASK_PRICE_43_diff,"ASK_PRICE_54_diff":ASK_PRICE_54_diff,
          "ASK_PRICE_65_diff":ASK_PRICE_65_diff,"ASK_PRICE_76_diff":ASK_PRICE_76_diff,
          "ASK_PRICE_87_diff":ASK_PRICE_87_diff,"ASK_PRICE_98_diff":ASK_PRICE_98_diff,
          "ASK_PRICE_109_diff":ASK_PRICE_109_diff,"BID_PRICE_21_diff":BID_PRICE_21_diff,
          "BID_PRICE_32_diff":BID_PRICE_32_diff,"BID_PRICE_43_diff":BID_PRICE_43_diff,
          "BID_PRICE_54_diff":BID_PRICE_54_diff,"BID_PRICE_65_diff":BID_PRICE_65_diff,
          "BID_PRICE_76_diff":BID_PRICE_76_diff,"BID_PRICE_87_diff":BID_PRICE_87_diff,
          "BID_PRICE_98_diff":BID_PRICE_98_diff,"BID_PRICE_109_diff":BID_PRICE_109_diff
          })
    return (v3)