#!/usr/bin/env python
#
# analyzing & plotting COVID-19 progress with pandas & gnuplot
#


# import importlib
# itertools = importlib.import_module('pandas')
from pandas.io.api import read_csv
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


def plotFunc(file, cols, n_max=None, logscale=True, ica=None, title=None):
    # n_min = 1 if logscale else 0
    n_min = 0
    n_max = n_max * 2 if logscale else n_max
    opts_ = opts
    opts_ = opts_ + 'set logscale y; \n' if logscale else opts_
    opts_ = opts_ + f'set yrange [{n_min}:{n_max}]; ' if n_max else opts_
    opts_ = opts_ + f'set title "{title}"; ' if title else opts_
    command = f'{opts_} plot for [i={cols}] "{file}" using 1:i with lp'
    command = command + f' , \'\' using 1:{ica} with lp' if ica else command
    subprocess.call(["gnuplot", "-p", "-e", command])


def plot_temp(label, n_head=5):
    df = read_csv(label, index_col=0).transpose()
    i_canada = df.index.get_loc('Canada')
    plotFunc(file=label,
             cols=f'2:{n_head+1}',
             n_max=df.max().max(),
             # logscale=False,
             ica=i_canada + 2,
             title=f"New {label} per 100K persons")


# plot_temp(label='out.csv', n_head=25)


def plot(label, n_head=5, normalize=False):
    if normalize:
        # df_normalized = read_csv(label + '-normalized.csv').transpose()
        df = read_csv(label + '-normalized-rolling.csv',
                      index_col=0).transpose()
        i_canada = df.index.get_loc('Canada')
        plotFunc(file=label + '-normalized-rolling.csv',
                 cols=f'2:{n_head+1}',
                 n_max=df.max().max(),
                 # logscale=False,
                 ica=i_canada + 2,
                 title=f"New {label} per 100K persons")
    else:
        df = read_csv(label + '-new.csv', index_col=0).transpose()
        i_canada = df.index.get_loc('Canada')
        plotFunc(file=label + '-new.csv',
                 cols=f'2:{n_head + 1}',
                 n_max=df.max().max(),
                 ica=i_canada + 2,
                 title=f"New {label}")
    return df


if len(q) > 0:
    n_max = int(q[0])
else:
    n_max = 10

df = plot(label='confirmed', n_head=n_max, normalize=True)
#df = plot(label='deaths', n_head=n_max, normalize=True)
df = plot(label='rate', n_head=n_max, normalize=True)
