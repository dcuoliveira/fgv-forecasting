import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.decomposition import PCA
import random
import pyfolio as pf

def simple_time_series_outofsample_predict(df, model, test_size):
    Y = pd.DataFrame(df[df.columns[0]])
    Y[Y >= 0] = 1
    Y[Y < 0] = -1
    X = df[df.columns[1:len(df.columns)]]
    listOut = []
    Y_train, Y_test, X_train, X_test, = train_test_split(Y, X, test_size=test_size, shuffle=False)
    fit_insample = model.fit(X_train, Y_train)
    df_predict_insample = Y_train.copy()
    df_predict_insample['predict'] = fit_insample.predict(X_train)
    for i in range(len(X_test)):
        X_train_in = X_train.append(X_test.iloc[0:i])
        Y_train_in = Y_train.append(Y_test.iloc[0:i])
        X_test_in = X_test.iloc[i:]
        fit = model.fit(X_train_in, Y_train_in)
        outofsample_predict = fit.predict(X_test_in)
        dict = {'date': Y_test.index[i], 'real': Y_test.iloc[i][Y.columns[0]], 'predict': outofsample_predict[0]}
        listOut.append(dict)
    df_rolling_outofsample = pd.DataFrame.from_dict(listOut)
    df_rolling_outofsample.index = df_rolling_outofsample['date']
    del df_rolling_outofsample['date']
    return df_predict_insample, df_rolling_outofsample

def curme_preis_stanley_moat_2013(df):
    dict_out = {}
    asset_return = pd.DataFrame(df[df.columns[0]])
    covariates = df[df.columns[1:len(df.columns)]]
    strategy_return = {}
    for col in covariates.columns:
        df_tmp = pd.DataFrame(covariates[col])
        df_tmp['rolling_mean'] = df_tmp.expanding(1).mean()
        df_tmp[col] = df_tmp[col] - df_tmp['rolling_mean']
        df_tmp = df_tmp.dropna()
        df_tmp[np.isneginf(df_tmp)] = 0
        df_tmp = df_tmp[[col]]
        for index, row in df_tmp.iterrows():
            if row.item() > 0:
               covariates.loc[index, col] = -1
            elif row.item() < 0:
                covariates.loc[index, col] = 1
            else:
                covariates.loc[index, col] = 0
        col_return = asset_return[asset_return.columns[0]] * covariates[col]
        strategy_return[col] = col_return
    df_out = pd.DataFrame.from_dict(strategy_return)
    df_out = pd.merge(asset_return, df_out, right_index=True, left_index=True)
    df_out = df_out.apply(lambda x: ((1 + x).cumprod()) - 1, axis=0)
    asset_return_copy = asset_return.copy()
    rnd_list = []
    random.seed(100)
    for i in range(len(asset_return)):
        rnd = random.sample([-1, 0, 1], 1)
        rnd_list.append(rnd[0])
    asset_return_copy['random_strat'] = asset_return_copy['log_pct_change'] * rnd_list
    asset_return_copy = asset_return_copy.apply(lambda x: ((1 + x).cumprod()) - 1, axis=0)
    df_out['random_strat'] = asset_return_copy['random_strat']
    return df_out


if __name__ == '__main__':
    # es = mng.connect_arctic()['BBG_EOD'].read('ES1 INDEX').data
    es = pd.read_csv(r"../data/es1.txt", sep='\t')
    es.index = pd.to_datetime(es['date'])
    es_close = es[['PX_LAST']]
    es_close = es_close.dropna()
    es_close = pd.DataFrame(es_close['PX_LAST'].resample(rule='W-MON').last())
    es_close['PX_LAST_s1'] = es_close['PX_LAST'].shift(+1)
    es_close['log_pct_change'] = np.log(es_close['PX_LAST'] / es_close['PX_LAST_s1'])
    del es_close['PX_LAST_s1']
    es_close = es_close.dropna()
    google = pd.read_csv('../../../../../QuantFin/google trends/dfOut_google_politics_normalized.csv')
    google.index = pd.to_datetime(google['date'])
    del google['date']
    google = google.asfreq('D')
    google_shift = google.shift(+1)
    google_shift = google_shift.dropna()

    df = pd.merge(es_close['log_pct_change'], google_shift, right_index=True, left_index=True)
    df = df.loc[:, (df != 0).any(axis=0)]

    ### Curme, Preis, Stanley and Moat 2013 ###
    df1 = curme_preis_stanley_moat_2013(df)
    random = df1['random_strat']
    df1 = df1.drop(columns=['random_strat'])
    print(np.median(df1.iloc[len(df1)-1]))
    print(np.mean(df1.iloc[len(df1) - 1]))
    print(np.std(df1.iloc[len(df1) - 1]))
    print(np.median(random))
    print(np.mean(random))
    print(np.std(random))
    plt.hist(df1.iloc[len(df1)-1], bins=10)
    plt.hist(random, bins=10)
    plt.legend(labels=['curme', 'random'])
    plt.show()

    ### Huang, Rojas and Convery 2019 ###
    # cols_to_subset = list(set(Gprediction_series) & set(cointegrated_series))
    cols_to_subset = list(pd.Series(Gprediction_series + cointegrated_series).unique())
    X_to_go = X5[cols_to_subset]
    df = pd.merge(Y, X_to_go, right_index=True, left_index=True)
    df_return_out = {}
    df_predict_out = {}
    for type in ['l1', 'l2']:
        print(type)
        for c in np.linspace(0, 1, 100):
            if c == 0 or c == 1:
                continue
            lasso_model = linear_model.LogisticRegression(penalty=type, C=c, solver='saga')
            lasso_out = simple_time_series_outofsample_predict(df, lasso_model, 0.7)
            df_lasso_in = lasso_out[0]
            df_lasso_out = lasso_out[1]
            df_lasso_tot = pd.concat([df_lasso_in, df_lasso_out])
            df_lasso_tot = pd.merge(es_close, df_lasso_tot['predict'], right_index=True, left_index=True)
            df_lasso_tot[type + '_return'] = df_lasso_tot['log_pct_change'] * df_lasso_tot['predict']
            df_lasso_tot[type + '_return'] = ((1 +  df_lasso_tot[type + '_return']).cumprod()) - 1
            df_return_out[type + '_' + str(c)] = df_lasso_tot[type + '_return']
            df_predict_out[type + '_' + str(c)] = df_lasso_tot['predict']
    df_return_out = pd.DataFrame.from_dict(df_return_out)
    df_return_out.to_csv('Z:/desenv/gtrends/df_return_es_l1_l2_augmented.csv')
    # df_return_out.plot()
    # plt.show()
    df_predict_out = pd.DataFrame.from_dict(df_predict_out)
    df_predict_out.to_csv('Z:/desenv/gtrends/df_predict_es_l1_l2_augmented.csv')
    series = df_predict_out[df_predict_out.columns[0]] * df['log_pct_change']
    series.index = pd.to_datetime(series.index)
    pf.create_returns_tear_sheet(series)
