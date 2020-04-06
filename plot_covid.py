#!/usr/bin/env python
#
# analyzing & plotting COVID-19 progress with pandas & gnuplot
#


import pandas as pd
import subprocess

opts = 'set key autotitle columnheader;' +\
       'set datafile separator comma;' +\
       'set key outside; set autoscale;' +\
       'set timefmt "%m/%d/%y";' +\
       'set xdata time;' +\
       'set xrange ["3/01/20":]; \n'
pop = pd.read_csv('pop', index_col='country')
n=15

def plotFunc(file, cols, n_max=None, logscale=True, i_canada=None, title=None):
    opts_ = opts
    opts_ = opts_ + 'set logscale y; \n' if logscale else opts_
    opts_ = opts_ + f'set yrange [:{n_max}]; ' if n_max else opts_
    opts_ = opts_ + f'set title "{title}"; ' if title else opts_
    command = f'{opts_} plot for [i={cols}] "{file}" using 1:i with lp'
    command = command + f' , \'\' using 1:{i_canada} with lp' if i_canada else command
    subprocess.call(["gnuplot", "-p", "-e", command])


def get_new(name, n_head=5):
    file = 'COVID-19/csse_covid_19_data/csse_covid_19_time_series/' + \
           'time_series_covid19_' + name + '_global.csv'

    file = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/' + \
            'master/csse_covid_19_data/csse_covid_19_time_series/' + \
            f'time_series_covid19_{name}_global.csv'
    print(f'url: <{file}>')

    df = pd.read_csv(file)
    df = df.drop(columns=['Lat', 'Long', 'Province/State'])
    df = df.set_index(['Country/Region'])

    # CALC NEW
    df_new = df.copy()
    for i in range(2, df.shape[1]):
        df_new.iloc[:, i] = df.iloc[:, i] - df.iloc[:, i - 1]
    df_new = df_new.groupby(level='Country/Region').sum()
    df_new = df_new.sort_values(by=df_new.columns[-1], ascending=False)
    return df_new


def normalizePop(label, df):
    df.index.name = 'country'
    mer = pd.merge(df, pop, how='right', left_on='country', right_on='country')

    mer_sum = mer.iloc[:,-2-n:-2].sum(axis=1)
    pop_ = mer['pop']
    mer_avg = 100000 * mer_sum/mer['pop']

    df = pd.DataFrame()
    df[f'{n}-day sum'] = mer_sum
    df[f'{n}-day/100K'] = mer_avg
    df = df[pop_ > 10000000]
    df = df.sort_values(by=f'{n}-day/100K', ascending=False)
    df.round(2).to_csv(label + '-summary.csv')

    mer = mer.drop(columns=['pop'])
    for c in mer.columns:
        mer[c] = 100000*mer[c]/pop_
    mer[f'{n}-day/100K'] = mer_avg/n
    mer = mer[pop_>10000000].dropna()
    mer = mer.round(2).sort_values(by=f'{n}-day/100K', ascending=False)

    return mer


def clean_and_plot(label, n_head=5, plot=True, normalize=False, rolling=1):
    df = get_new(label, n_head=n_head)
    df.transpose().round(2).to_csv(label+'-new.csv')
    if plot:
        i_canada = df.index.get_loc('Canada')
        plotFunc(file=label+'-new.csv',
                 cols=f'2:{n_head+1}',
                 n_max=df.max().max(),
                 i_canada=i_canada+2,
                 title=f"New {label}")

    if normalize:
        df_normalized = normalizePop(label, df)
        df_normalized.transpose().to_csv(label + '-normalized.csv')

        df_rolling = df_normalized.rolling(window=rolling).mean().dropna()
        df_rolling.transpose().round(2).to_csv(label + '-normalized-rolling.csv')

        i_canada = df_rolling.index.get_loc('Canada')
        if plot:
            plotFunc(file=label+'-normalized-rolling.csv',
                     cols=f'2:{n_head+1}',
                     n_max=df_rolling.max().max(),
                     logscale=False,
                     i_canada=i_canada+2,
                     title=f"New {label} per 100K persons")
    return df

df = clean_and_plot(label='confirmed', n_head=10, plot=True, normalize=True, rolling=3)
df = clean_and_plot(label='Deaths', n_head=10, plot=True, normalize=True, rolling=3)
