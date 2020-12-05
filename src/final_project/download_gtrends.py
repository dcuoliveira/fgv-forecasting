from pytrends.request import TrendReq
import pandas as pd

kw_list = ["debt","color","stocks","restaurant","portfolio","inflation","housing","dow jones","revenue","economics","credit","markets","return","unemployment","money","religion","cancer","growth","investment","hedge","marriage","bonds","derivatives","headlines","profit","society","leverage","loss","cash","office","fine","stock","market","banking","crisis","happy","car","nasdaq","gains","finance","sell","invest","fed","house","metals","travel","returns","gain","default","present","holiday","water","rich","risk","gold","success","oil","war","economy","DOW JONES","chance","short","sell","lifestyle","greed","food","financial","markets","movie","nyse","ore","BUY AND HOLD","opportunity","health","short","selling","earnings","arts","culture","bubble","buy","trader","rare","earths","tourism","politics","energy","consume","consumption","freedom","dividend","world","conflict","kitchen","forex","home","crash","transaction","garden","fond","train","labor","fun","environment","ring"]
dfList = []
kw_list = list(pd.Series(kw_list).unique())
for kw in kw_list:
    print(kw)
    list_to_search = [kw]
    list_to_search.append('google')
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(kw_list=list_to_search, geo='US')
    df = pytrends.interest_over_time()
    del df['isPartial']
    del df['google']
    dfList.append(df)
dfOut = pd.concat(dfList, axis=1)
dfOut.to_csv('dfOut_google_politics_normalized.csv')
