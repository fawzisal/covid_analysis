#!/usr/bin/env python
#
# analyzing & plotting COVID-19 progress in Ontario with pandas & gnuplot
#


import pandas as pd
import argparse
import os
import subprocess as sp

os.makedirs('raw', exist_ok=True)
parser = argparse.ArgumentParser(description='Download, clean and plot COVID-19 data')
parser.add_argument('-d', action='store_true',
                    help='download new hospitalization data',
                    dest='b_d', default=False)
parser.add_argument('-p', action='store_true', help='plot hospitalization indicators',
                    dest='b_p', default=False)
parser.add_argument('-a', action='store_true', help='plot all hospitalization indicators',
                    dest='b_a', default=False)
parser.add_argument('-D', action='store_true',
                    help='download new case data [per health region]',
                    dest='b_D', default=False)
parser.add_argument('-P', action='store_true', help='plot new cases [per health region]',
                    dest='b_P', default=False)
args = parser.parse_args()

opts = 'set key font "arial";' +\
       'set key autotitle columnheader;' +\
       'set datafile separator comma;' +\
       'set key outside; set autoscale;' +\
       'set timefmt "%m/%d/%y";' +\
       'set xdata time;' +\
       'set xrange ["3/01/20":]; \n' +\
       'toggle all; \n'

opts_DT = 'set key font "arial";' +\
          'set key autotitle columnheader;' +\
          'set datafile separator comma;' +\
          'set key outside; set autoscale;' +\
          'set timefmt "%Y-%m-%d";' +\
          'set xdata time;' +\
          'set xrange ["2020-03-01":]; \n' +\
          'set format x "%m-%d"; \n' +\
          'toggle all; \n'


def plotFunc(file, n_head=10, n_max=None, logscale=False,
             n_min=0, format='lp',
             ica=[], title=None, df=None):
    n_min = 0.01 if logscale else n_min
    opts_ = opts
    opts_ = opts_ + 'set logscale y; \n' if logscale else opts_
    opts_ = opts_ + f'set yrange [{n_min}:{n_max}]; ' if n_max else opts_
    opts_ = opts_ + f'set title "{title}"; ' if title else opts_
    opts_ = opts_ + f'set xzeroaxis; '
    command = f'{opts_} plot for [i=2:{n_head + 1}] "{file}" using 1:i with {format}'

    for ican in ica:
        if ican - 2 > n_head:
            command = command + f' , \'\' using 1:{ican} with {format}'
    sp.call(["gnuplot", "-p", "-e", command])


def plotFunc_DT(file, n_head=10, n_max=None, logscale=False,
                n_min=0, format='lp',
                ica=[], title=None, df=None):
    n_min = 0.01 if logscale else n_min
    opts_ = opts_DT
    opts_ = opts_ + 'set logscale y; \n' if logscale else opts_
    opts_ = opts_ + f'set yrange [{n_min}:{n_max}]; ' if n_max else opts_
    opts_ = opts_ + f'set title "{title}"; ' if title else opts_
    opts_ = opts_ + f'set xzeroaxis; '
    command = f'{opts_} plot for [i=2:{n_head + 1}] "{file}" using 1:i with {format}'

    for ican in ica:
        if ican - 2 > n_head:
            command = command + f' , \'\' using 1:{ican} with {format}'
    sp.call(["gnuplot", "-p", "-e", command])


def casesF(df, index='Age_Group', columns='Reporting_PHU'):
    return df.pivot_table(columns=columns, index=index,
                          values='Outcome', aggfunc='count')


def fatalF(df, index='Age_Group', columns='Reporting_PHU'):
    return df.pivot_table(columns=columns, index=index,
                          values='Outcome', aggfunc='sum')


def pivot(df, index='Age_Group', columns='Reporting_PHU', trim=True):
    fatal = fatalF(df, columns=columns, index=index)
    cases = casesF(df, columns=columns, index=index)
    fatality = (100 * fatal.fillna(0) / cases).fillna(0).round(3)
    if trim:
        threshold = df.shape[0] / 100
        fatality = fatality[fatality.columns[cases.sum() > threshold]]
        fatality = fatality[cases.sum(axis=1) > threshold]
    return fatality


if args.b_D:
    url = 'https://data.ontario.ca/dataset/f4112442-bdc8-45d2-be3c-12efae72fb27/resource/455fd63b-603d-4608-8216-7d8647f43350/download/conposcovidloc.csv'
    print(f'url: <{url}>')
    df_D = pd.read_csv(url, index_col=0)
    df_D.to_csv('raw_ON/conposcovidloc.csv')

if args.b_P:
    df = pd.read_csv('raw_ON/conposcovidloc.csv', index_col=0).fillna(0)
    df['Reporting_PHU'] = df['Reporting_PHU'].str.extract('^(\w+)\W')
    df_pivot = df.pivot_table(index='Accurate_Episode_Date',
                              columns='Reporting_PHU',
                              aggfunc=pd.Series.count,
                              values='Outcome1').fillna(0).astype(int).iloc[1:]
    df_pivot['total'] = df_pivot.sum(axis=1)
    df_pivot.loc['total'] = df_pivot.iloc[1:].sum()
    df_pivot = df_pivot.sort_values(axis=1, by=df_pivot.index[-1], ascending=False)
    df_pivot = df_pivot.sort_index(ascending=False)
    df_pivot.to_csv('raw_ON/cases_pivot.csv')
    df_pivot.iloc[:, 1:] = (100 * df_pivot.iloc[:, 1:]).div(df_pivot.iloc[:, 0], axis=0).astype(int)
    df_pivot.to_csv('raw_ON/cases_pivot_per.csv')
    df_perpop = df.pivot_table(index='Accurate_Episode_Date',
                               columns='Reporting_PHU',
                               aggfunc=pd.Series.count,
                               values='Outcome1').fillna(0).astype(int).iloc[1:]
    pop = pd.read_csv('pop_ON.csv')
    df_perpop.loc['total'] = df_perpop.iloc[1:].sum()
    df_perpop = df_perpop.sort_values(axis=1,
                                      by=df_perpop.index[-1], ascending=False)
    df_perpop = df_perpop.div((pop['pop'] / 1e5).to_list()).astype(int)
    df_perpop = df_perpop.sort_index(ascending=False)
    df_perpop.to_csv('raw_ON/cases_perpop.csv')

    plotFunc_DT(file='raw_ON/cases_perpop.csv', title="New cases per 100K people",
                df=df_pivot, n_head=10, logscale=True)

    df.Accurate_Episode_Date = pd.to_datetime(df.Accurate_Episode_Date)
    df.Case_Reported_Date = pd.to_datetime(df.Case_Reported_Date)
    df.Test_Reported_Date = pd.to_datetime(df.Test_Reported_Date)
    df.Specimen_Date = pd.to_datetime(df.Specimen_Date)
    df.Outbreak_Related = df.Outbreak_Related == 'Yes'
    df['Outcome'] = df.Outcome1 == 'Fatal'
    df['month'] = pd.to_datetime(df.Accurate_Episode_Date).dt.month
    dfm = df[['Accurate_Episode_Date', 'Case_Reported_Date',
              'Test_Reported_Date', 'Specimen_Date', 'Age_Group',
              'Client_Gender', 'Case_AcquisitionInfo', 'Outbreak_Related',
              'Reporting_PHU', 'Outcome']]
    dfm.columns = ['EpiDT', 'CaseDT', 'TestDT', 'SpcDT', 'Age', 'Gender',
                   'Acqui', 'Outbreak', 'PHU', 'Death']

if args.b_d:
    url = 'https://data.ontario.ca/dataset/f4f86e54-872d-43f8-8a86-3892fd3cb5e6/resource/ed270bb8-340b-41f9-a7c6-e8ef587e6d11/download/covidtesting.csv'
    print(f'url: <{url}>')
    df_D = pd.read_csv(url, index_col=0)
    df_D.to_csv('raw_ON/covidtesting.csv')

if args.b_p:
    df = pd.read_csv('raw_ON/covidtesting.csv', index_col=0).fillna(0)
    df.columns = ['conf-', 'assum-', 'assum+', 'conf+', 'resolved',
                  'deaths', 'total', 'total approved for testing',
                  'num tests', 'test_perc+', 'under investigation',
                  'num hospitalized', 'num icu', 'num on ventilator',
                  'ltc resident+', 'ltc hcw+', 'ltc resident deaths',
                  'ltc hcw deaths']
    df.index = pd.to_datetime(df.index).strftime('%-m/%-d/%y')
    dfo = df
    df.to_csv('raw_ON/data.csv')
    dfh = dfo.iloc[:, 10:14]
    dfh.iloc[:, 0] = dfh.iloc[:, 0]
    dfh.iloc[:, 1] = dfh.iloc[:, 1]
    dfh.columns = ['under investigation/1000', '# hospitalized/10',
                   '# icu', '# ventilator']
    dfh.to_csv('raw_ON/data_hos.csv')
    dfd = dfo
    dfd = dfd.rolling(window=7, axis=1, min_periods=7).mean().round(2)
    dfd = dfd.diff(periods=1, axis=1).dropna(axis=1)
    dfd = dfd.dropna()
    dfd = 100 * dfd.div(dfd.max())
    dfd.to_csv('raw_ON/data_diff.csv')
    n = 60
    if args.b_a:
        plotFunc(file='raw_ON/data.csv', title="Detailed COVID-19 indicators for Ontario",
                 df=df, n_head=17, logscale=True)
    else:
        plotFunc(file='raw_ON/data_hos.csv', title="Detailed COVID-19 indicators for Ontario",
                 df=df, n_max=dfh.max().max(),
                 logscale=True, n_head=4)
    # plotFunc(file='raw_ON/data_diff.csv', df=df, n_max=120, n_head=17)
