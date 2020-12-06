from pytrends.request import TrendReq
import pandas as pd
import os

kw_list = ["debt","color","stocks","restaurant","portfolio","inflation","housing","dow jones","revenue","economics","credit","markets","return","unemployment","money","religion","cancer","growth","investment","hedge","marriage","bonds","derivatives","headlines","profit","society","leverage","loss","cash","office","fine","stock","market","banking","crisis","happy","car","nasdaq","gains","finance","sell","invest","fed","house","metals","travel","returns","gain","default","present","holiday","water","rich","risk","gold","success","oil","war","economy","DOW JONES","chance","short","sell","lifestyle","greed","food","financial","markets","movie","nyse","ore","BUY AND HOLD","opportunity","health","short","selling","earnings","arts","culture","bubble","buy","trader","rare","earths","tourism","politics","energy","consume","consumption","freedom","dividend","world","conflict","kitchen","forex","home","crash","transaction","garden","fond","train","labor","fun","environment","ring"]
dfList = []
kw_list = list(pd.Series(kw_list).unique())

for kw in kw_list:
    print(kw)

    init_dt = '2005-01-01'
    kw_list = []
    for end_dt in ['2010-01-01', '2015-01-01', '2020-01-01']:
        list_to_search = [kw]
        list_to_search.append('google')
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(kw_list=list_to_search, geo='US', timeframe=init_dt + ' ' + end_dt)
        df = pytrends.interest_over_time()
        kw_list.append(df)
        init_dt = end_dt
        del df['isPartial']
        del df['google']
    dfList.append(pd.concat(kw_list))
dfOut = pd.concat(dfList, axis=1)
dfOut.to_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', 'dfOut_google_normalized.csv'))
