from rename_column import *

###Function to create v2 and v5 
def createV2V5():
	data = rename_data()
 	#create v2
  	ASK_BID_PRICE_11_diff =data['ASK_PRICE1']-data['BID_PRICE1']
  	ASK_BID_PRICE_22_diff =data['ASK_PRICE2']-data['BID_PRICE2']
	ASK_BID_PRICE_33_diff =data['ASK_PRICE3']-data['BID_PRICE3']
	ASK_BID_PRICE_44_diff =data['ASK_PRICE4']-data['BID_PRICE4']
	ASK_BID_PRICE_55_diff =data['ASK_PRICE5']-data['BID_PRICE5']
	ASK_BID_PRICE_66_diff =data['ASK_PRICE6']-data['BID_PRICE6']
	ASK_BID_PRICE_77_diff =data['ASK_PRICE7']-data['BID_PRICE7']
	ASK_BID_PRICE_88_diff =data['ASK_PRICE8']-data['BID_PRICE8']
	ASK_BID_PRICE_99_diff =data['ASK_PRICE9']-data['BID_PRICE9']
	ASK_BID_PRICE_1010_diff =data['ASK_PRICE10']-data['BID_PRICE10']


	ASK_BID_PRICE_11_mean = (data['ASK_PRICE1']+data['BID_PRICE1'])/2
	ASK_BID_PRICE_22_mean = (data['ASK_PRICE2']+data['BID_PRICE2'])/2
	ASK_BID_PRICE_33_mean = (data['ASK_PRICE3']+data['BID_PRICE3'])/2
	ASK_BID_PRICE_44_mean = (data['ASK_PRICE4']+data['BID_PRICE4'])/2
	ASK_BID_PRICE_55_mean = (data['ASK_PRICE5']+data['BID_PRICE5'])/2
	ASK_BID_PRICE_66_mean = (data['ASK_PRICE6']+data['BID_PRICE6'])/2
	ASK_BID_PRICE_77_mean = (data['ASK_PRICE7']+data['BID_PRICE7'])/2
	ASK_BID_PRICE_88_mean = (data['ASK_PRICE8']+data['BID_PRICE8'])/2
	ASK_BID_PRICE_99_mean = (data['ASK_PRICE9']+data['BID_PRICE9'])/2
	ASK_BID_PRICE_1010_mean = (data['ASK_PRICE10']+data['BID_PRICE10'])/2

	v2 = pd.DataFrame({ 'ASK_BID_PRICE_11_diff' : ASK_BID_PRICE_11_diff,
		'ASK_BID_PRICE_22_diff' : ASK_BID_PRICE_22_diff,
		'ASK_BID_PRICE_33_diff' : ASK_BID_PRICE_33_diff,
		'ASK_BID_PRICE_44_diff' : ASK_BID_PRICE_44_diff,
		'ASK_BID_PRICE_55_diff' : ASK_BID_PRICE_55_diff,
		'ASK_BID_PRICE_66_diff' : ASK_BID_PRICE_66_diff,
		'ASK_BID_PRICE_77_diff' : ASK_BID_PRICE_77_diff,
		'ASK_BID_PRICE_88_diff' : ASK_BID_PRICE_88_diff,
		'ASK_BID_PRICE_99_diff' : ASK_BID_PRICE_99_diff,
		'ASK_BID_PRICE_1010_diff' : ASK_BID_PRICE_1010_diff,
		'ASK_BID_PRICE_11_mean' : ASK_BID_PRICE_11_mean,
		'ASK_BID_PRICE_22_mean' : ASK_BID_PRICE_22_mean,
		'ASK_BID_PRICE_33_mean' : ASK_BID_PRICE_33_mean,
		'ASK_BID_PRICE_44_mean' : ASK_BID_PRICE_44_mean,
		'ASK_BID_PRICE_55_mean' : ASK_BID_PRICE_55_mean,
		'ASK_BID_PRICE_66_mean' : ASK_BID_PRICE_66_mean,
		'ASK_BID_PRICE_77_mean' : ASK_BID_PRICE_77_mean,
		'ASK_BID_PRICE_88_mean' : ASK_BID_PRICE_88_mean,
		'ASK_BID_PRICE_99_mean' : ASK_BID_PRICE_99_mean,
		'ASK_BID_PRICE_1010_mean' : ASK_BID_PRICE_1010_mean})



	#create v5
	ASK_BID_PRICE_diff_sum =(data['ASK_PRICE1']-data['BID_PRICE1'])+ \
	(data['ASK_PRICE2']-data['BID_PRICE2'])+ \
	(data['ASK_PRICE3']-data['BID_PRICE3'])+ \
	(data['ASK_PRICE4']-data['BID_PRICE4'])+ \
	(data['ASK_PRICE5']-data['BID_PRICE5'])+ \
	(data['ASK_PRICE6']-data['BID_PRICE6'])+ \
	(data['ASK_PRICE7']-data['BID_PRICE7'])+ \
	(data['ASK_PRICE8']-data['BID_PRICE8'])+ \
	(data['ASK_PRICE9']-data['BID_PRICE9'])+ \
	(data['ASK_PRICE10']-data['BID_PRICE10'])


	ASK_BID_SIZE_diff_sum =(data['ASK_SIZE1']-data['BID_SIZE1'])+ \
	(data['ASK_SIZE2']-data['BID_SIZE2'])+ \
	(data['ASK_SIZE3']-data['BID_SIZE3'])+ \
	(data['ASK_SIZE4']-data['BID_SIZE4'])+ \
	(data['ASK_SIZE5']-data['BID_SIZE5'])+ \
	(data['ASK_SIZE6']-data['BID_SIZE6'])+ \
	(data['ASK_SIZE7']-data['BID_SIZE7'])+ \
	(data['ASK_SIZE8']-data['BID_SIZE8'])+ \
	(data['ASK_SIZE9']-data['BID_SIZE9'])+ \
	(data['ASK_SIZE10']-data['BID_SIZE10'])

	v5 = pd.DataFrame({ 'ASK_BID_PRICE_diff_sum' : ASK_BID_PRICE_diff_sum,
		'ASK_BID_SIZE_diff_sum' : ASK_BID_SIZE_diff_sum})

	#combine v1 and v4 
  	v2_v5 = pd.concat([v2,v5], axis=1)
  	return(v2_v5)