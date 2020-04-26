#!/usr/bin/env python3
from scipy import optimize
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


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


def curve(effects, time):
  x, t, s = effects, time, 0
  for i in range(0, len(x), 3):
    a, f, p = x[i : i+3]
    s += a*1900*np.tanh(0.5*f*t + p)
  return s

def loss(effects, actuals):
  x, s, e = effects, actuals, 0
  for t in range(0, len(s)):
    e += (s[t] - curve(x, t)) ** 2
  # print(e)
  return e


def main(actuals, effects):
  x = actuals
  fig = plt.figure()
  plt.plot(range(0, len(x)), x[::])
  x = np.repeat(x, 2)
  for i in range(0, len(x)):
    x[i] = curve(effects, i)
  plt.plot(range(0, len(x)), x[::])
  plt.show()


def merge_provinces(rows):
  dates = list(set(rows['Date']))
  if len(dates) == len(rows): return rows
  rows = rows.copy()
  for date in dates:
    is_date = rows['Date'] == date
    rowd, value = rows[is_date], 0
    for i in range(0, len(rowd)):
      value += rowd['Value']
    rows.loc[is_date, 'Value'] = value
  province0 = rows['Province/State'].iloc[0]
  is_province0 = rows['Province/State'] == province0
  return rows[is_province0]


def get_country(rows, country, merge=True):
  is_country = rows['Country/Region'] == country
  rows = rows[is_country]
  return merge_provinces(rows) if merge else rows


csv = 'time_series_covid19_confirmed_global_narrow.csv'
rows = pd.read_csv(csv, comment='#')
rows = get_country(rows, 'Italy')
print(rows)
vals = list(reversed(list(rows['Value'])))

x0 = np.ones(6)
x1 = optimize.minimize(lambda x: loss(x, vals), x0)
print(x1.x)
main(vals, x1.x)
