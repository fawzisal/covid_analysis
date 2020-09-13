#!/usr/bin/env python
#
# analyzing & plotting COVID-19 progress with pandas & gnuplot
#


# import importlib
# itertools = importlib.import_module('pandas')
from pandas.io.api import read_csv
from numpy import where
import subprocess
import sys

q = sys.argv[1:]

opts = 'set key font "arial";' +\
       'set key autotitle columnheader;' +\
       'set datafile separator comma;' +\
       'set key outside; set autoscale;' +\
       'set timefmt "%m/%d/%y";' +\
       'set xdata time;' +\
       'set xrange ["3/01/20":]; \n' +\
       'toggle all; \n'
n = 15
countries = ["Canada"]
countries = ["Canada", "US", "United Kingdom", "Sweden",
             "Spain", "Italy", "Germany", "Iran", "China",
             "Korea S."]


def plotFunc(file, n_head=10, n_max=None, logscale=True,
             ica=[], title=None, df=None):
    n_min = 0.01 if logscale else 0
    # n_min = 0
    n_max = n_max * 2 if logscale else n_max
    n_max = min(n_max, 100)
    opts_ = opts
    opts_ = opts_ + 'set logscale y; \n' if logscale else opts_
    opts_ = opts_ + f'set yrange [{n_min}:{n_max}]; ' if n_max else opts_
    opts_ = opts_ + f'set title "{title}"; ' if title else opts_
    command = f'{opts_} plot for [i=2:{n_head + 1}] "{file}" using 1:i with lp'
    for ican in ica:
        if ican - 2 > n_head:
            command = command + f' , \'\' using 1:{ican} with lp'
    subprocess.call(["gnuplot", "-p", "-e", command])


def plot_temp(label, n_head=5, normalize=False):
    df = read_csv(label + '-normalized-rolling.csv', index_col=0).transpose()
    # i_canada = df.index.get_loc('Canada')
    plotFunc(file=label + '-normalized-rolling.csv',
             n_head=n_head,
             n_max=df.max().max(),
             # logscale=False,
             ica=where(df.index.isin(countries))[0] + 2,
             title=f"Fatality {label} (%)",
             df=df)


# plot_temp(label='out.csv', n_head=25)


def plot(label, n_head=5, normalize=False):
    if normalize:
        # df_normalized = read_csv(label + '-normalized.csv').transpose()
        df = read_csv(label + '-normalized-rolling.csv',
                      index_col=0).transpose()
        # i_canada = df.index.get_loc('Canada')
        plotFunc(file=label + '-normalized-rolling.csv',
                 n_head=n_head,
                 n_max=df.max().max(),
                 # logscale=False,
                 ica=where(df.index.isin(countries))[0] + 2,
                 title=f"New {label} per 100K persons",
                 df=df)
    else:
        df = read_csv(label + '-new.csv', index_col=0).transpose()
        # i_canada = df.index.get_loc('Canada')
        plotFunc(file=label + '-new.csv',
                 n_head=n_head,
                 n_max=df.max().max(),
                 ica=where(df.index.isin(countries))[0] + 2,
                 title=f"New {label}",
                 df=df)
    return df


if len(q) > 0:
    n_max = int(q[0])
else:
    n_max = 20

# df = plot(label='confirmed', n_head=n_max, normalize=True)
df = plot(label='deaths', n_head=n_max, normalize=True)
# df = plot_temp(label='rate', n_head=n_max, normalize=True)
