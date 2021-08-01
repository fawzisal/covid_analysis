#!/usr/bin/env python
#
# analyzing & plotting COVID-19 progress with pandas & gnuplot
#


import pandas as pd
import numpy as np
import os
import subprocess
import argparse
import datetime as dt


os.makedirs('raw_ON', exist_ok=True)
parser = argparse.ArgumentParser(description='Download, analyze and plot COVID data')
parser.add_argument('-i', action='store_true', help='print info',
                    dest='b_i', default=False)
parser.add_argument('-d', action='store_true', help='download new data',
                    dest='b_d', default=False)
parser.add_argument('-k', action='store_true', help='clean downloaded data',
                    dest='b_k', default=False)
parser.add_argument('-P', nargs=1, help='period to average data over (default: 7)',
                    dest='period', default=7)
parser.add_argument('-p', nargs='?',
                    help='choose functions to plot [c: confirmed, d: deaths, a: active, r: rate, f: diff, R: recovered] (default: cd)',
                    dest='plots', default='cd')
parser.add_argument('-c', nargs='?',
                    help='countries to include by default (default: ca,us,uk,fr,de,it,es,ne,tr,in,ch,ko)',
                    dest='countries', default='ca,us,uk,fr,de,it,es,nl,tr,in,ch,kr')
parser.add_argument('-r', nargs='?',
                    help='regions to Africa: FC/FE/FS/FW, America: MC/MN/MS, Asia: AC/AE/AI/AS/AW, Europe: EE/EN/ES/EW, MidEast: MA/MG/ML, Oceania: OC',
                    dest='regions', default='')
parser.add_argument('-n', nargs=1,
                    help='number of countries to plot (default: 5)',
                    dest='num', default=5)
parser.add_argument('-s', action='store_true', help='sort by cases',
                    dest='b_s', default=False)
args = parser.parse_args()


if args.b_i:
    # TODO: prettyprint info
    pass

pop_base0 = pd.read_csv('pop.csv')
pop_base = pop_base0.set_index('country')
pop = pop_base.copy()
pop = pop.iloc[:, :1]
pop['ccode'] = pop.index.to_series().astype(str)
country_code = pop_base.index.to_series(index=pop_base.country_code)
region_codes = pop_base0.groupby('code').country.agg(lambda x: list(x))

opts = """
        set key font "arial";
        set key autotitle columnheader;
        set datafile separator comma;
        set key outside; set autoscale;
        set timefmt "%m/%d/%y";
        set xdata time;
        set xtics format "%b";
        set mxtics (4);
        set grid;
        toggle all;
"""

n = args.period
regions = region_codes.reindex(args.regions.upper().split(',')).sum()
countries = country_code.reindex(args.countries.upper().split(',')).to_list()
regions = [] if not regions else regions
countries = regions + countries


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

    df.to_csv('raw/' + label + '.csv')

    # CALC NEW
    df_new = df.copy()
    for i in range(2, df.shape[1]):
        df_new.iloc[:, i] = df.iloc[:, i] - df.iloc[:, i - 1]
    df_new = df_new.groupby(level='Country/Region').sum()
    df_new = df_new.sort_values(by=df_new.columns[-1], ascending=False)
    return df_new


def dl(label):
    df = get_new(label)
    df.to_csv('raw/' + label + '-raw.csv')
    return df


def sort_top(df, n_head=5):
    df = df.transpose()
    df_sel = df.iloc[0:n_head]
    df_nonsel = df.iloc[n_head:]
    df_sel['tot'] = df_sel.iloc[-n:].median(axis=1)
    df_sel = df_sel.sort_values(by='tot', ascending=False)
    return pd.concat([df_sel, df_nonsel], sort=False, axis=0).transpose()


def sort_all(df, n_head=5):
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
    df[f'total per 100K'] = all_avg
    df[f'{n}-day total'] = mer_sum
    df[f'{n}-day total per 100K'] = mer_avg
    df[f'daily avg per 100K'] = mer_sum / n
    df = df[pop_ > 1e7]
    df = df.sort_values(by=f'daily avg per 100K', ascending=False)
    sort_all(df.round(4), n_head).to_csv('raw/' + label + '-summary.csv')

    mer = mer.sort_values(by='pop', ascending=False)
    mer = mer[mer['pop'] > 1e7].dropna()
    mer = mer.drop(columns=['pop'])
    for c in mer.columns:
        mer[c] = 1e5 * mer[c] / pop_
    mer[f'{n}-day/100K'] = mer_avg / n

    return mer, df


def cleanRate(label, normalize=False, rolling=1, tosort=False):
    conf = pd.read_csv('raw/confirmed-raw.csv', index_col=0)
    death = pd.read_csv('raw/deaths-raw.csv', index_col=0)
    df = (100 * death / conf).round(4)
    df.to_csv(f'raw/{label}-raw.csv')
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
        df_rolling.transpose().to_csv(f'raw/{label}-normalized-rolling.csv')
    return df_rolling


def cleanDiff(label, normalize=False, rolling=1, tosort=False):
    df = pd.read_csv('raw/confirmed-new.csv', index_col=0).transpose()
    df.loc[pop.index, 'pop'] = pop.loc[pop.index, 'pop']
    df = 1e5 * df.div(df['pop'], axis=0).dropna().iloc[:, :-1]
    dft = df
    df = df.rolling(window=7, axis=1, min_periods=7).mean()
    df = df.diff(periods=1, axis=1).dropna(axis=1)
    df = df.iloc[:, ::7].round(2)
    df = df.sort_values(by=df.columns[-1], ascending=False)
    dfs = df / df.abs().fillna(0)
    df = (df * dfs).apply(np.sqrt) * dfs
    df.transpose().to_csv('raw/confirmed-diff.csv')
    return df, dft


def clean(label, n_head=5, normalize=False, rolling=1, tosort=False):
    df = pd.read_csv('raw/' + label + '-raw.csv', index_col=0)
    df_sorted = df.transpose().round(4)
    if tosort:
        df_sorted = sort_all(df_sorted, n_head)
    df_sorted.to_csv('raw/' + label + '-new.csv')

    if normalize:
        df_normalized, df_summary = normalizePop(label, df, n_head)
        if tosort:
            df_sorted = sort_all(df_normalized.transpose(), n_head)
        else:
            df_sorted = df_normalized.transpose()

        df_sorted.to_csv('raw/' + label + '-normalized.csv')

        df_rolling = df_normalized.rolling(window=rolling, axis=1,
                                           min_periods=1)
        df_rolling = df_rolling.mean().dropna().transpose().round(4)
        if tosort:
            df_rolling = sort_all(df_rolling, n_head)

        df_rolling.to_csv('raw/' + label + '-normalized-rolling.csv')

        return df_summary
    return df


def plotFunc(file, n_head=10, num=None, logscale=True,
             n_min=0, format='lp',
             ica=[], title=None, df=None):
    n_min = 0.01 if logscale else n_min
    num = num * 2 if logscale else num
    opts_ = opts
    opts_ = opts_ + 'set xrange ["3/01/20":"' + dt.date.today().strftime('%m/%d/%y') + '"]; \n'
    opts_ = opts_ + 'set logscale y; \n' if logscale else opts_
    opts_ = opts_ + f'set yrange [{n_min}:{num}]; ' if num else opts_
    opts_ = opts_ + f'set title "{title}"; ' if title else opts_
    opts_ = opts_ + f'set xzeroaxis; '
    command = f'{opts_} plot for [i=2:{n_head + 1}] "{file}" using 1:i with {format}'

    for ican in ica:
        if ican - 2 > n_head:
            command = command + f' , \'\' using 1:{ican} with {format}'
    subprocess.call(["gnuplot", "-p", "-e", command])


def plot_rate(label, n_head=5, normalize=False):
    df = pd.read_csv('raw/' + label +
                     '-normalized-rolling.csv', index_col=0).transpose()
    # n_mean = df.mean().mean()
    # n_std = df.mean().std()
    s = df.fillna(-1).to_numpy().flatten()
    s = s[s > -1]

    plotFunc(file='raw/' + label + '-normalized-rolling.csv',
             n_head=n_head,
             n_min=np.quantile(s, 0.01),
             num=np.quantile(s, 0.99),
             # n_min=max(n_mean - 2 * n_std, df.min().min(), 0),
             # num=n_mean + 3 * n_std,
             logscale=True,
             ica=np.where(df.index.isin(countries))[0] + 2,
             title=f"Fatality {label} (%)",
             df=df)
    return df


def plot_diff(label, n_head=5, normalize=False):
    file = 'raw/' + label + '-diff.csv'
    df = pd.read_csv(file, index_col=0).transpose()
    title = f"New {label} [diff]"

    n_mean = df.mean().mean()
    n_std = df.mean().std()

    plotFunc(file=file,
             n_head=n_head,
             n_min=n_mean - 20 * n_std,
             num=n_mean + 20 * n_std,
             format='lp',
             ica=np.where(df.index.isin(countries))[0] + 2,
             title=title,
             logscale=False,
             df=df)
    return df


def plot(label, n_head=5, normalize=False):
    if normalize:
        postfix = '-normalized-rolling.csv'
        df = pd.read_csv('raw/' + label + postfix, index_col=0).transpose()
        title = f"New {label} per 100K persons"
    else:
        postfix = '-new.csv'
        df = pd.read_csv('raw/' + label + postfix, index_col=0).transpose()
        title = f"New {label}"

    plotFunc(file='raw/' + label + postfix,
             n_head=n_head,
             num=df.max().max(),
             ica=np.where(df.index.isin(countries))[0] + 2,
             title=title,
             df=df)
    return df


if args.b_d:
    df_confirmed = dl(label='confirmed')
    df_deaths = dl(label='deaths')
    df_recovered = dl(label='recovered')
    df_active = df_confirmed - df_recovered
    df_active.to_csv('raw/active-raw.csv')

rolling = n
if args.b_k:
    tosort = args.b_s
    num = 5
    df_conf = clean(label='confirmed', n_head=num, tosort=tosort,
                    normalize=True, rolling=rolling)
    df_sum = (df_conf.iloc[:, 3] / 25).astype(int).apply(lambda x: "-" * x)
    df_sum = pd.DataFrame(df_sum)
    df_sum.merge(pop_base.iloc[:, 1:2],
                 left_index=True,
                 right_index=True,
                 how='left').to_csv('raw/conf_summary.csv')
    df_death = clean(label='deaths', n_head=num, tosort=tosort,
                     normalize=True, rolling=rolling)
    df_recovered = clean(label='recovered', n_head=num, tosort=tosort,
                         normalize=True, rolling=rolling)
    df_active = clean(label='active', n_head=num, tosort=tosort,
                      normalize=True, rolling=rolling)
    df_rate = cleanRate(label='rate', tosort=tosort,
                        normalize=True, rolling=rolling)
    df_rate.iloc[:, -30:].mean(axis=1).round(4).to_csv('raw/rate-summary.csv',
                                                       header=True)
    df_diff, dft = cleanDiff(label='confirmed')

num = int(args.num[0] if type(args.num) is list else args.num)
if args.plots:
    if "c" in args.plots:
        df = plot(label='confirmed', n_head=num, normalize=True)
    if "d" in args.plots:
        df = plot(label='deaths', n_head=num, normalize=True)
    if "R" in args.plots:
        df = plot(label='recovered', n_head=num, normalize=True)
    if "a" in args.plots:
        df = plot(label='active', n_head=num, normalize=True)
    if "r" in args.plots:
        df = plot_rate(label='rate', n_head=num, normalize=True)
    if "f" in args.plots:
        df = plot_diff(label='confirmed', n_head=num - 5, normalize=True)

if not (args.b_d or args.b_k or args.plots):
    parser.print_help()
