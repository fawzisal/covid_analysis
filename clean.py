#!/usr/bin/env python
#
# analyzing & plotting COVID-19 progress with pandas & gnuplot
#


import pandas as pd
import sys

q = sys.argv[1:]

opts = 'set key autotitle columnheader;' +\
       'set datafile separator comma;' +\
       'set key outside; set autoscale;' +\
       'set timefmt "%m/%d/%y";' +\
       'set xdata time;' +\
       'set xrange ["3/01/20":]; \n'
pop = pd.read_csv('pop', index_col='country')
n = 180


def sort_top(df, n_head=5):
    return pd.concat([df.iloc[:, 0:n_head].sort_index(axis=1),
                      df.iloc[:, n_head:]], sort=False, axis=1)


def sort_all(df):
    df['tot'] = df.sum(axis=1)
    return df.sort_values(by='tot', ascending=False).iloc[:, :-1]

def normalizePop(label, df, n_head=5):
    df.index.name = 'country'
    mer = pd.merge(df, pop, how='right', left_on='country', right_on='country')
    mer_ = mer.iloc[:, :-2]

    mer_sum = mer_.iloc[:, -n:].sum(axis=1)
    all_sum = mer_.sum(axis=1)
    pop_ = mer['pop']
    mer_avg = 1e5 * mer_sum / pop_
    all_avg = 1e5 * all_sum / pop_

    df = pd.DataFrame()
    df[f'total'] = all_sum
    # df[f'death rate%'] = 0.365 * all_avg / ((mer_ != 0).sum(axis=1))
    df[f'total per 100K'] = all_avg
    df[f'{n}-day total'] = mer_sum
    df[f'{n}-day total per 100K'] = mer_avg
    df[f'daily avg per 100K'] = mer_sum / n
    # df[f'yearly avg'] = 0.365 * mer_avg / n
    df = df[pop_ > 10000000]
    df = df.sort_values(by=f'daily avg per 100K', ascending=False)
    sort_top(df.round(2), n_head).to_csv(label + '-summary.csv')

    mer = mer.drop(columns=['pop'])
    for c in mer.columns:
        mer[c] = 100000 * mer[c] / pop_
    mer[f'{n}-day/100K'] = mer_avg / n
    mer = mer[pop_ > 10000000].dropna()
    mer = mer.round(2).sort_values(by=f'{n}-day/100K', ascending=False)

    return mer, df


def cleanRate(label, normalize=False, rolling=1):
    conf = pd.read_csv('confirmed-raw.csv', index_col=0)
    death = pd.read_csv('deaths-raw.csv', index_col=0)
    df = (100 * death / conf).round(2)
    df.to_csv(f'{label}-raw.csv')
    if normalize:
        df_rolling = df.rolling(window=rolling, axis=1, min_periods=1).mean()
        df_rolling['tot'] = death.sum(axis=1)
        df_rolling = df_rolling.sort_values('tot', ascending=False)
        df_rolling = df_rolling.drop(columns='tot')
        df_rolling.transpose().to_csv(f'{label}-normalized-rolling.csv')
    return df_rolling

def clean(label, n_head=5, normalize=False, rolling=1):
    df = pd.read_csv(label + '-raw.csv', index_col=0)
    sort_top(df.transpose().round(2), n_head).to_csv(label + '-new.csv')

    if normalize:
        df_normalized, df_summary = normalizePop(label, df, n_head)
        sort_top(df_normalized.transpose(), n_head).to_csv(label + '-normalized.csv')

        df_rolling = df_normalized.rolling(window=rolling, axis=1, min_periods=1).mean().dropna()
        sort_top(df_rolling.transpose().round(2), n_head).to_csv(label + '-normalized-rolling.csv')
        return df_summary
    return df


if len(q) > 0:
    rolling = int(q[0])
else:
    rolling = 3

n_max = 5
df_conf = clean(label='confirmed', n_head=n_max, normalize=True, rolling=rolling)
df_death = clean(label='deaths', n_head=n_max, normalize=True, rolling=rolling)
df_rate = cleanRate(label='rate', normalize=True, rolling=rolling)
df_rate.iloc[:, -30:].mean(axis=1).round(3).to_csv('rate-summary.csv', header=True)
# df_rate.iloc[:, -30:].round(3).to_csv('rate-summary.csv', header=True)
