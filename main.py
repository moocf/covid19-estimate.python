#!/usr/bin/env python3
from scipy import optimize
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math

# fitting corona data:
# italy: total cases

# estimating curve:
# effect of public:    a1*tanh(f1*t + p1)
# ...       doctors:   a2*tanh(f2*t + p2)
# ...       engineers: a3*tanh(f3*t + p3)
# total cases = sum of effects

# loss function:
# mean square error

# optimize in batches with:
# - gradient descent
# - newtons method


def csv_read(file):
  rows = pd.read_csv(file, comment='#')
  rows.drop(['Lat', 'Long', 'ISO 3166-1 Alpha 3-Codes', 'Region Code',
    'Sub-region Code', 'Intermediate Region Code'], axis=1, inplace=True)
  return rows


def curve(x, cs):
  n, s = cs.shape[0], 0
  for i in range(0, n, 3):
    a, b, c = cs[i : i+3]
    s += a * math.exp(-((x-b)**2)/(2*c*c))
  return s


def loss(cs, y):
  e = 0
  for x in range(0, len(y)):
    e += (y[x] - curve(cs, x)) ** 2
  return e


def derivative_at(fn, cs, x, i, dc=1e-4):
  y0 = fn(cs, x)
  cs = cs.copy()
  cs[i] = cs[i]+dc
  y1 = fn(cs, x)
  return (y1 - y0) / dc


def gradient_at(fn, cs, x):
  n = cs.shape[0]
  a = np.zeros(n)
  for i in range(n):
    a[i] = derivative_at(fn, cs, x, i)
  return a


def gradient_descent(cs, y, l, max_iter=10000):
  for i in range(max_iter):
    print('loss', loss(cs, y), cs)
    dcs = -l * gradient_at(loss, cs, y)
    # if np.sum(np.abs(dcs)) < 1e-8: break
    cs  = cs + dcs
  return cs


def main(actuals, effects):
  x = actuals
  fig = plt.figure()
  plt.plot(range(0, len(x)), x[::])
  x = np.repeat(x, 2)
  for i in range(0, len(x)):
    x[i] = curve(effects, i)
  plt.plot(range(0, len(x)), x[::])
  plt.show()


def filter_country(rows, country):
  is_country = rows['Country/Region'] == country
  return rows[is_country]


def merge_date(rows):
  a = rows.iloc[0:0]
  dates = sorted(set(rows['Date']))
  for d in dates:
    rd = rows[rows['Date'] == d]
    rv = rd.iloc[0:1].copy()
    rv['Value'] = rd['Value'].sum()
    a = a.append(rv)
  return a


def diff_value(rows):
  r, c = rows.shape
  v0, value = 0, rows.columns.get_loc('Value')
  for i in range(r):
    v1 = rows.iloc[i, value]
    rows.iloc[i, value] = v1 - v0
    v0 = v1
  return rows


def average_value(rows, window=7):
  r, c = rows.shape
  w, value = [0] * window, rows.columns.get_loc('Value')
  for i in range(r):
    v1 = rows.iloc[i, value]
    w.append(v1)
    w.pop(0)
    rows.iloc[i, value] = sum(w) / window
  return rows


csvfile = 'time_series_covid19_confirmed_global_narrow.csv'
rows = csv_read(csvfile)
rows = filter_country(rows, 'Italy')
rows = merge_date(rows)
rows = diff_value(rows)
rows = average_value(rows, 14)
vals = list(rows['Value'])

cs = np.asarray([1000, 0, 1])
l = np.asarray([1, 1, 1])
cs = gradient_descent(cs, vals, l)
print(cs)
# x0 = np.ones(6)
# x1 = optimize.minimize(lambda x: loss(x, vals), x0)
# print(x1.x)
# main(vals, x1.x)
