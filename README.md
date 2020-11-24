
A little code to analyze and plot the progress of COVID-19: new cases normalized by population and averaged over a 7-day period.

# Requirements

- python (required packages: pandas, numpy)
- gnuplot
- *tested on OSX 20.1.0*

# Usage

- For COVID-19 data per country: `python covid.py -h` or `./covid.py -h`

The following code downloads new data, cleans up data, and plots the time series for cases and deaths in Canada, US, UK, Germany and Netherlands:

> ./covid.py -dk -p cd -c ca,us,uk,de,ne

- For COVID-19 data in Ontario: `python oncovid.py -h` or `./oncovid.py -h`

The following code download, analyzes and plots the latest data.

> ./oncovid.py -dDpP
