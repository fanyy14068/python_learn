"""
create by fanyy on 10/26/2018
S-H-ESD methods for time series anomaly detection
step:
1, stl decomposition, let s = ts - ts.median() - seasonal
2, for i in 1:max_outerlies:
   do:
       R_i = max(|x-x_median| / sigma)
       λ_i = (n−i) * t_(p,n−i−1) / sqrt((n−i−1+t^2_(p,n−i−1))(n−i+1))
       where t stands for t-distribution, p is the quantile
       if R_i > λ_i, then i is an anomaly
warning:
if the length of data is two long, the effect seems not well

reference:
1, https://github.com/twitter/AnomalyDetection
2, https://github.com/nicolasmiller/pyculiarity
3, https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

"""
import time
import numpy as np
import pandas as pd
from rstl import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.robust.scale import mad
from scipy.stats import t as student_t

from machine_learning.time_series import date_util
import matplotlib.pylab as plt


def detect_anoms(data, k=0.49, alpha=0.05, num_obs_per_period=None, one_tail=True, upper_tail=True, time_interval=60):
    """
    # Detects anomalies in a time series using S-H-ESD.
    #
    # Args:
    #	 data: Time series to perform anomaly detection on.
    #	 k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
    #	 alpha: The level of statistical significance with which to accept or reject anomalies.
    #	 num_obs_per_period: Defines the number of observations in a single period, and used during seasonal decomposition.
    #	 one_tail: If TRUE only positive or negative going anomalies are detected depending on if upper_tail is TRUE or FALSE.
    #	 upper_tail: If TRUE and one_tail is also TRUE, detect only positive going (right-tailed) anomalies. If FALSE and one_tail is TRUE, only detect negative (left-tailed) anomalies.
    #	 time_interval: min time sample interval
    # Returns:
    #   A dictionary containing the anomalies (anoms) and decomposition components (stl).
    """

    num_obs = len(data)
    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    if num_obs < num_obs_per_period * 2:
        raise ValueError("Anom detection needs at least 2 periods worth of data")

    resample = '{}T'.format(int(time_interval / 60))
    ts = data.asfreq(resample).interpolate().dropna()

    # use seasonal_decompose instead of stl, seasonal_decompose is more faster  than stl
    decomp = seasonal_decompose(ts, model="additive", two_sided=False, freq=num_obs_per_period,
                                extrapolate_trend='freq')
    resids = data - decomp.seasonal - data.median()

    # decomp = STL(data.values, num_obs_per_period, 'periodic', robust=True)
    # resids = data - decomp.seasonal - data.median()

    n = len(resids)
    max_outliers = int(num_obs * k)
    if max_outliers == 0:
        return {}

    num_anoms = 0
    anoms_idx = [0] * max_outliers

    for i in range(1, max_outliers + 1):
        median = resids.median()
        ares = calc_ares(resids, median, one_tail, upper_tail)
        p = 1 - alpha / float(n - i + 1) if one_tail else 1 - alpha / float(2 * (n - i + 1))
        data_sigma = mad(resids.values)
        if data_sigma == 0:
            break

        ares = ares / float(data_sigma)
        R_i = ares.max()

        temp_max_idx = ares[ares == R_i].index.tolist()[0]
        anoms_idx[i - 1] = temp_max_idx

        t = student_t.ppf(p, (n - i - 1))
        lam = t * (n - i) / float(np.sqrt((n - i - 1 + t**2) * (n - i + 1)))

        if R_i > lam:
            num_anoms = i
            res_median = median
            threshold = resids[temp_max_idx]
        else:
            break

        resids = resids[resids.index != temp_max_idx]

    if num_anoms == 0:
        return {}
    else:
        anoms_idx = anoms_idx[:num_anoms]
        return {
            'anoms': [item for item in anoms_idx if item in data.index],
            'res_median': res_median,
            'data_median': data.median(),
            'threshold': threshold,
            'seasonal': decomp.seasonal[-num_obs_per_period:].tolist(),
            'seasonal_index': int(ts.index[-num_obs_per_period].value / 1e9),
            'direction': [one_tail, upper_tail],
        }


def calc_ares(resids, median, one_tail, upper_tail):
    if one_tail:
        if upper_tail:
            ares = resids - median
        else:
            ares = median - resids
    else:
        ares = np.abs(resids - median)
    return ares


def df_to_series(df):
    return pd.Series(df['value'].values, index=df['timestamp'], dtype=np.float64)


def detect_anoms_online(ts, model, time_interval=60, num_obs_per_period=1440):
    if not model:
        return []
    resample = '{}T'.format(int(time_interval / 60))
    ts_without_na = ts.asfreq(resample).interpolate().dropna()

    test_start_time = ts_without_na.index[0].value / 1e9
    period_cur = int((test_start_time - model['seasonal_index']) / time_interval) % num_obs_per_period
    resids = (ts_without_na - np.asarray(model['seasonal'][period_cur: period_cur+len(ts_without_na)]) -
              model['data_median'])

    one_tail, upper_tail = model['direction']
    ares = calc_ares(resids, model['res_median'], one_tail, upper_tail)
    threshold = calc_ares(model['threshold'], model['res_median'], one_tail, upper_tail)

    anoms_idx = ts_without_na[ares > threshold].index
    return [item for item in anoms_idx if item in ts.index]


def plot_result(ts_train, ts_test, model, anoms):
    ts_train.plot()
    ts_test.plot(color='g')
    if len(anoms):
        ts_test[anoms].plot(color='r', marker='o', linestyle='')

    if model and model['anoms']:
        ts_train[model['anoms']].plot(color='r', marker='o', linestyle='')
    plt.show()


def main():
    df = pd.read_csv('./TestData/ts_data1.csv', usecols=['timestamp', 'value', 'label'])
    df.timestamp = date_util.format_time_for_timestamp(df.timestamp)

    ts = df_to_series(df[-4032:])
    ts_train = ts[:-288]
    ts_test = ts[-288:]

    result = detect_anoms(ts_train, k=0.01, num_obs_per_period=288, one_tail=False, upper_tail=False, time_interval=300)
    anoms = detect_anoms_online(ts_test, result, time_interval=300, num_obs_per_period=288)
    # plot_result(ts_train, ts_test, result, anoms)


if __name__ == '__main__':
    main()
