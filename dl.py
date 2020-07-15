#!/usr/bin/env python
#
# analyzing & plotting COVID-19 progress with pandas & gnuplot
#


import pandas as pd


def get_new(label):
    file = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/' + \
           'time_series_covid19_' + label + '_global.csv'

    file = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/' + \
           'master/csse_covid_19_data/csse_covid_19_time_series/' + \
           f'time_series_covid19_{label}_global.csv'
    print(f'url: <{file}>')

    df = pd.read_csv(file)
    df = df.drop(columns=['Lat', 'Long', 'Province/State'])
    df = df.set_index(['Country/Region'])

    df.to_csv(label + '.csv')
    # CALC NEW
    df_new = df.copy()
    for i in range(2, df.shape[1]):
        df_new.iloc[:, i] = df.iloc[:, i] - df.iloc[:, i - 1]
    df_new = df_new.groupby(level='Country/Region').sum()
    df_new = df_new.sort_values(by=df_new.columns[-1], ascending=False)
    return df_new


def dl(label):
    df = get_new(label)
    df.to_csv(label + '-raw.csv')


df = dl(label='confirmed')
df = dl(label='deaths')
