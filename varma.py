from sklearn import datasets
import numpy as np 
import math
import sys
import pickle
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf


use_viewport = 1
use_object = 0


dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
offset = int(sys.argv[4])
usernum=6


data = []
frame_nos = []


# ds 1
width=3840.0
height=1920.0
view_width = 3840.0
view_height = 2048.0
milisec = 1.0

# # ds 2
# width=2560.0
# height=1280.0
# view_width = 2560.0
# view_height = 1440.0
# milisec = 1.0

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



obj_info = np.load('Obj_traj/ds{}/ds{}_topic{}.npy'.format(dataset, dataset, topic), allow_pickle=True,  encoding='latin1').item()
view_info = pickle.load(open('Viewport/ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, usernum), 'rb'), encoding='latin1')


n_objects = []
for i in obj_info.keys():
	try:
		n_objects.append(max(obj_info[i].keys()))
	except:
		n_objects.append(0)
total_objects = max(n_objects)

max_frame = int(view_info[-1][0]*1.0*fps/milisec)

for i in range(len(view_info)-1):
	frame = int(view_info[i][0]*1.0*fps/milisec)
	frame += int(offset*1.0*fps/milisec)

	frame_nos.append(frame)
	if(frame > max_frame):
		break
	X={}
	X['VIEWPORT_x']=int(view_info[i][1][0]*width/view_width)
	X['VIEWPORT_y']=int(view_info[i][1][1]*height/view_height)
	for j in range(total_objects):
		try:
			centroid = obj_info[frame][j]

			if obj_info[frame][j] == None:
				X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
				X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)
			else:
				X['OBJ_'+str(j)+'_x']=centroid[0]
				X['OBJ_'+str(j)+'_y']=centroid[1]

		except:
			X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
			X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)


	data.append((X, int(view_info[i+1][1][0]*width/view_width),int(view_info[i+1][1][1]*height/view_height)))



series = []
for line in data:
	if use_viewport == 1:
		a = [-1 for i in range(2)]
	if use_object == 1:
		a = [-1 for i in range(total_objects)]

	for key, value in line[0].items():
		if use_viewport == 1:
			if key == 'VIEWPORT_x':
				a[0] = value
			if key == 'VIEWPORT_y':
				a[1] = value
		if use_object == 1:
			if key != 'VIEWPORT_x' and key != 'VIEWPORT_y':
				a.append(value)

	series.append(a)

series = np.array(series, dtype=np.float64)



a = 230
b = 260



result = adfuller(series[a:b, 0])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


series_log = np.log(series[a:b, 0])
series_x = np.diff(series_log, 1)
# df_series_log = pd.DataFrame({'X':series_log[:]})
# df_rolling_mean = df_series_log.rolling(window=2).mean()
# rolling_mean = df_rolling_mean.to_numpy().reshape(-1)
# series_x = series_log - rolling_mean
series_x = series_x[~np.isnan(series_x)]

result = adfuller(series_x)
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


print('##########################')

result = adfuller(series[a:b, 1])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


series_log = np.log(series[a:b, 1])
series_y = np.diff(series_log, 1)
# df_series_log = pd.DataFrame({'X':series_log[:]})
# df_rolling_mean = df_series_log.rolling(window=2).mean()
# rolling_mean = df_rolling_mean.to_numpy().reshape(-1)
# series_y = series_log - rolling_mean
series_y = series_y[~np.isnan(series_y)]

result = adfuller(series_y)
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))




# plt.figure()
# plt.plot([i for i in range(len(series_x))], series_x, color='b')
# plt.plot([i for i in range(len(series_y))], series_y, color='r')

# plt.figure()
# acf_x = acf(series_x)
# plt.plot([i for i in range(len(acf_x))], acf_x)
# plt.title("ACF_x")

# plt.figure()
# acf_y = acf(series_y)
# plt.plot([i for i in range(len(acf_y))], acf_y)
# plt.title("ACF_y")


# plt.figure()
# pacf_x = pacf(series_x)
# plt.plot([i for i in range(len(pacf_x))], pacf_x)
# plt.title("PACF_x")

# plt.figure()
# pacf_y = pacf(series_y)
# plt.plot([i for i in range(len(pacf_y))], pacf_y)
# plt.title("PACF_y")

# plt.show()


series_new = []
for i in range(len(series_x)):
	series_new.append([series_x[i], series_y[i]])


model_x = SARIMAX(series_x, order=(1,1,0))
model_fit_x = model_x.fit(maxiter=1000, disp=0)
model_pred_x = model_fit_x.forecast(30)

model_y = ARIMA(series_y, order=(5,0,0))
model_fit_y = model_y.fit(maxiter=1000, disp=0)
model_pred_y = model_fit_y.forecast(30)[0]

print(model_pred_x)

# model = VARMAX(series_new, order=(4,0), trend='ct')
# model_fit = model.fit(maxiter=1000, disp=0)
# z = model_fit.forecast(30)


act_x, act_y = series[b-1:b+30, 0], series[b-1:b+30, 1]
# x_pred = []
# for i in range(len(z)):
# 	if i == 0:
# 		x_pred.append(np.exp(z[i, 0]) * act_x[i])
# 	else:
# 		x_pred.append(np.exp(z[i, 0]) * x_pred[i-1])
# 	print(act_x[i+1], x_pred[i])


# print("###")
# y_pred = []
# for i in range(len(z)):
# 	if i == 0:
# 		y_pred.append(np.exp(z[i, 1]) * act_y[i])
# 	else:
# 		y_pred.append(np.exp(z[i, 1]) * y_pred[i-1])
# 	print(act_y[i+1], y_pred[i])



x_pred_list, y_pred_list = [], []
for k in range(len(model_pred_x)):
	if k == 0:
		x_pred_list.append(np.exp(model_pred_x[k]) * act_x[i])
		y_pred_list.append(np.exp(model_pred_y[k]) * act_y[i])
	else:
		x_pred_list.append(np.exp(model_pred_x[k]) * x_pred_list[k-1])
		y_pred_list.append(np.exp(model_pred_y[k]) * y_pred_list[k-1])