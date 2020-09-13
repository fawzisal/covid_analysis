#!/usr/bin/env python
#
# analyzing & plotting COVID-19 progress with pandas & gnuplot
#


import pandas as pd
import sys
import argparse

q = sys.argv[1:]

opts = 'set key autotitle columnheader;' +\
       'set datafile separator comma;' +\
       'set key outside; set autoscale;' +\
       'set timefmt "%m/%d/%y";' +\
       'set xdata time;' +\
       'set xrange ["3/01/20":]; \n'
pop_base = pd.read_csv('pop', index_col='country')
pop = pop_base.iloc[:, :1]
# pop.index = pop_base.code + '-' + pop_base.index
# pop['ccode'] = pop_base.code + '-' + pop_base.index
pop['ccode'] = '' + pop_base.index
n = 60


parser = argparse.ArgumentParser(description='boredom reliever')
parser.add_argument('query', metavar='QUERY', type=str, nargs='*',
                    help='the question to answer')
parser.add_argument('-f', '--file', help='the list of words to bin',
                    default="", type=str)
parser.add_argument('-n', '--num', help='the level of abstraction',
                    default=2, type=int)
parser.add_argument('-p', '--print', help='the level of abstraction',
                    default=1, type=int)
args = vars(parser.parse_args())

n = args['num']
f = args['file']
p = args['print']


def sort_top(df, n_head=5):
    df = df.transpose()
    df_sel = df.iloc[0:n_head]
    df_nonsel = df.iloc[n_head:]
    df_sel['tot'] = df_sel.iloc[-n:].median(axis=1)
    df_sel = df_sel.sort_values(by='tot', ascending=False)
    # df_sel = df_sel.iloc[:-1]
    return pd.concat([df_sel, df_nonsel], sort=False, axis=0).transpose()
    # df = df.sort_values(by='tot', ascending=False).iloc[:, :-1].transpose()
    # return df


def sort_all(df, n_head=5):
    # df = df.transpose()
    # df['tot'] = df.iloc[-n:].mean(axis=1)
    # return df.sort_values(by='tot', ascending=False).transpose()
    return df.sort_values(by=df.index[-1], ascending=False, axis=1)


def normalizePop(label, df, n_head=5):
    df.index.name = 'country'
    mer = pd.merge(df, pop, how='right', left_on='country', right_on='country')
    mer.index = mer['ccode']
    mer = mer.iloc[:, :-1]
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
    sort_all(df.round(4), n_head).to_csv(label + '-summary.csv')

    mer = mer.sort_values(by='pop', ascending=False)
    mer = mer[mer['pop'] > 10000000].dropna()
    mer = mer.drop(columns=['pop'])
    for c in mer.columns:
        mer[c] = 100000 * mer[c] / pop_
    mer[f'{n}-day/100K'] = mer_avg / n
    # mer = mer.round(4).sort_values(by=f'{n}-day/100K', ascending=False)

    return mer, df


def cleanRate(label, normalize=False, rolling=1, tosort=False):
    conf = pd.read_csv('confirmed-raw.csv', index_col=0)
    death = pd.read_csv('deaths-raw.csv', index_col=0)
    df = (100 * death / conf).round(4)
    df.to_csv(f'{label}-raw.csv')
    mer = pd.merge(df, pop, how='right',
                   left_index=True,
                   right_on='country')
    if not tosort:
        mer = mer.sort_values(by='pop', ascending=False)
        mer = mer.drop(columns=['pop'])
    if normalize:
        df_rolling = mer.rolling(window=rolling, axis=1, min_periods=1).mean()
        df_rolling['tot'] = death.sum(axis=1)
        if tosort:
            df_rolling = df_rolling.sort_values('tot', ascending=False)
        df_rolling = df_rolling.drop(columns='tot')
        df_rolling.transpose().to_csv(f'{label}-normalized-rolling.csv')
    return df_rolling


def clean(label, n_head=5, normalize=False, rolling=1, tosort=False):
    df = pd.read_csv(label + '-raw.csv', index_col=0)
    df_sorted = df.transpose().round(4)
    if tosort:
        df_sorted = sort_all(df_sorted, n_head)
    df_sorted.to_csv(label + '-new.csv')

    if normalize:
        df_normalized, df_summary = normalizePop(label, df, n_head)
        if tosort:
            df_sorted = sort_all(df_normalized.transpose(), n_head)
        else:
            df_sorted = df_normalized.transpose()

        df_sorted.to_csv(label + '-normalized.csv')

        df_rolling = df_normalized.rolling(window=rolling, axis=1,
                                           min_periods=1).mean().dropna().transpose().round(4)
        if tosort:
            df_rolling = sort_all(df_rolling, n_head)

        df_rolling.to_csv(label + '-normalized-rolling.csv')

        return df_summary
    return df


if len(q) > 0:
    rolling = int(q[0])
else:
    rolling = 7

tosort = True
n_max = 5
df_conf = clean(label='confirmed', n_head=n_max, tosort=tosort,
                normalize=True, rolling=rolling)
df_death = clean(label='deaths', n_head=n_max, tosort=tosort,
                 normalize=True, rolling=rolling)
df_rate = cleanRate(label='rate', tosort=tosort,
                    normalize=True, rolling=rolling)
df_rate.iloc[:, -30:].mean(axis=1).round(4).to_csv('rate-summary.csv',
                                                   header=True)
# df_rate.iloc[:, -30:].round(3).to_csv('rate-summary.csv', header=True)
