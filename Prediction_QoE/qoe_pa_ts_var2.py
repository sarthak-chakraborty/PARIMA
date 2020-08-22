from creme import linear_model
from creme import compose
from creme import compat
from creme import metrics
from creme import model_selection
from creme import optim
from creme import preprocessing
from creme import stream
from sklearn import datasets
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np 
import pandas as pd
import math
import sys
import pickle
import random
import warnings
from sklearn import neural_network as nn

dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
offset=int(sys.argv[4])
pref_quality = sys.argv[5]
fps_fraction = float(sys.argv[6])
usernum=2
ncol_tiles=8
nrow_tiles=8
pred_nframe=int(fps*fps_fraction)

bitrates = {'360p':1, '480p':2.5, '720p':5, '1080p':8, '1440p':16}	# [360p, 480p, 720p, 1080p, 1440p]
player_width = 600
player_height = 300
pref_bitrate = bitrates[pref_quality]

# # OUR DATA
# width=3840.0
# height=1920.0
# view_width = 1280
# view_height = 720
# milisec = 1000.0

# ds 1
width=3840.0
height=1920.0
view_width = 3840.0
view_height = 2048.0
milisec = 1.0

# ds 2
# width=2560.0
# height=1280.0
# view_width = 2560.0
# view_height = 1440.0
# milisec = 1.0
player_tiles_x = math.ceil(player_width*ncol_tiles*1.0/width)
player_tiles_y = math.ceil(player_height*nrow_tiles*1.0/height)
nn_model = nn.MLPRegressor(hidden_layer_sizes=(),activation='relu',verbose=False,warm_start=True)
nn_model.fit([[0,0,0,0]],[[0,0]])
nn_model.coefs_=[np.array([[0.25,0.25],[0.25,0.25],[0.25,0.25],[0.25,0.25]])]

def get_data(data, frame_nos, dataset, topic, usernum):

	
	obj_info = np.load('Obj_traj/ds{}/ds{}_topic{}.npy'.format(dataset, dataset, topic), allow_pickle=True,  encoding='latin1').item()
	view_info = pickle.load(open('Viewport/ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, usernum), 'rb'), encoding='latin1')


	n_objects = []
	for i in obj_info.keys():
		try:
			n_objects.append(max(obj_info[i].keys()))
		except:
			n_objects.append(0)
	total_objects=max(n_objects)

	max_frame = int(view_info[-1][0]*1.0*fps/milisec)

	for i in range(len(view_info)-1):
		frame = int(view_info[i][0]*1.0*fps/milisec)
		frame += int(offset*1.0*fps/milisec)

		frame_nos.append(frame)
		if(frame > max_frame):
			break
		X={}
		X['VIEWPORT_x']=int(view_info[i][1][1]*width/view_width)
		X['VIEWPORT_y']=int(view_info[i][1][0]*height/view_height)
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


		data.append((X, int(view_info[i+1][1][1]*width/view_width),int(view_info[i+1][1][0]*height/view_height)))
	
	return data, frame_nos, max_frame,total_objects


def pred_frames(data, model, metric_X, metric_Y, frames, prev_frames, tile_manhattan_error, act_tiles, pred_tiles, count,total_error1,total_error2):
	x_pred, y_pred = 0,0
	series = []
	shift_x = False

	for k in range(len(prev_frames)):
		[inp_k, x_act, y_act] = data[prev_frames[k]]
		a = [-1 for i in range(2)]
		for key, value in inp_k.items():
			if key == 'VIEWPORT_x':
				a[0] = value
			if key == 'VIEWPORT_y':
				a[1] = value

		if k==0:
			series.append(a)
			continue

		[prev_x,prev_y] = series[-1]
		
		if prev_x < a[0]:
			new_x_dif = a[0]-width
			if a[0] - prev_x > prev_x - new_x_dif:
				a[0] = new_x_dif
		else:
			new_x_dif = a[0]+width
			if prev_x - a[0] > new_x_dif - prev_x:
				a[0] = new_x_dif

		if a[0]<0:
			shift_x = True

		series.append(a)

	if shift_x == True:
		for i in range(len(series)):
			series[i][0] = series[i][0]+width

	series = np.array(series, dtype=np.float64)
	series_copy=series

	result = len(series[:, 0]) > 0 and all(elem == series[0][0] for elem in series[:, 0])
	if(result):
		series[:, 0] = [elem + random.random()/10 for elem in series[:, 0]]
	for i in range(len(series[:, 0])):
		if series[i, 0] == 0:
			series[i, 0] += random.random()
	series_log = np.log(series[:, 0])
	series_x = np.diff(series_log, 1)

	result = len(series[:, 1]) > 0 and all(elem == series[0][1] for elem in series[:, 1])
	if(result):
		series[:, 1] = [elem + random.random()/10 for elem in series[:, 1]]
	for i in range(len(series[:, 1])):
		if series[i, 1] == 0:
			series[i, 1] += random.random()
	series_log = np.log(series[:, 1])
	series_y = np.diff(series_log, 1)

	series_new = []
	for i in range(len(series_x)):
		series_new.append([series_x[i], series_y[i]])


	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		
		model_x = SARIMAX(series_x, order=(2,0,1))
		model_fit_x = model_x.fit(maxiter=1000, disp=0, method='nm')
		model_pred_x = model_fit_x.forecast(len(frames))

		model_y = SARIMAX(series_y, order=(3,0,0))
		model_fit_y = model_y.fit(maxiter=1000, disp=0, method='nm')
		model_pred_y = model_fit_y.forecast(len(frames))

	# print(model_pred_x)
	x_pred_list_ts, y_pred_list_ts = [], []
	x_pred_list_par, y_pred_list_par = [], []
	data_act=[]
	data_inp=[]

	for k in range(len(model_pred_x)):
		if k == 0:
			x_pred_list_ts.append(np.exp(model_pred_x[k]) * series_copy[-1][0])
			y_pred_list_ts.append(np.exp(model_pred_y[k]) * series_copy[-1][1])
		else:
			x_pred_list_ts.append(np.exp(model_pred_x[k]) * x_pred_list_ts[k-1])
			y_pred_list_ts.append(np.exp(model_pred_y[k]) * y_pred_list_ts[k-1])

	for k in range(len(frames)):
		[inp_k, x_act, y_act] = data[frames[k]]
		data_act.append([x_act,y_act])

		if(shift_x==True):
			x_pred_list_ts[k] = x_pred_list_ts[k] - width

		if(k == 0):
			x_pred, y_pred = model.predict_one(inp_k, True, x_act, y_act)
			x_pred_list_par.append(x_pred)
			y_pred_list_par.append(y_pred)
		else:
			inp_k['VIEWPORT_x'] = x_pred
			inp_k['VIEWPORT_y'] = y_pred
			x_pred, y_pred = model.predict_one(inp_k, True, None, None)	
			x_pred_list_par.append(x_pred)
			y_pred_list_par.append(y_pred)	

		[[x_pred,y_pred]] = nn_model.predict([[x_pred_list_ts[k],y_pred_list_ts[k],x_pred_list_par[k],y_pred_list_par[k]]])
		data_inp.append([x_pred_list_ts[k],y_pred_list_ts[k],x_pred_list_par[k],y_pred_list_par[k]])

		shift = 0
		if(x_act > x_pred):
			if(abs(x_act - x_pred) > abs(x_act - (x_pred+width))):
				x_pred = x_pred+width
				shift = 1
		else:
			if(abs(x_act - x_pred) > abs(x_act - (x_pred-width))):
				x_pred = x_pred-width
				shift = 2

		metric_X = metric_X.update(x_act, x_pred)
		metric_Y = metric_Y.update(y_act, y_pred)

		if(shift == 0):
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int(x_pred * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)
		elif(shift == 1):
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int((x_pred - width) * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)
		else:
			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			pred_tile_col = int((x_pred + width) * ncol_tiles / width)
			pred_tile_row = int(y_pred * nrow_tiles / height)

		actual_tile_row = actual_tile_row-nrow_tiles if(actual_tile_row >= nrow_tiles) else actual_tile_row
		actual_tile_col = actual_tile_col-ncol_tiles if(actual_tile_col >= ncol_tiles) else actual_tile_col
		actual_tile_row = actual_tile_row+nrow_tiles if actual_tile_row < 0 else actual_tile_row
		actual_tile_col = actual_tile_col+ncol_tiles if actual_tile_col < 0 else actual_tile_col

		######################################################
		# print("x: "+str(x_act))
		# print("x_pred: "+str(x_pred))
		# print("y: "+str(y_act))	
		# print("y_pred: "+str(y_pred))
		# print("("+str(actual_tile_row)+","+str(actual_tile_col)+"),("+str(pred_tile_row)+","+str(pred_tile_col)+")")
		# ######################################################
		
		act_tiles.append((actual_tile_row, actual_tile_col))
		pred_tiles.append((pred_tile_row, pred_tile_col))

		tile_col_dif = ncol_tiles
		tile_row_dif = actual_tile_row - pred_tile_row

		if actual_tile_col < pred_tile_col:
			tile_col_dif = min(pred_tile_col - actual_tile_col, actual_tile_col + ncol_tiles - pred_tile_col)
		else:
			tile_col_dif = min(actual_tile_col - pred_tile_col, ncol_tiles + pred_tile_col - actual_tile_col)


		current_tile_error = abs(tile_row_dif) + abs(tile_col_dif)
		if(current_tile_error <= (player_tiles_x+player_tiles_y)/2):
			current_tile_error = 0
		else:
			current_tile_error -= (player_tiles_x+player_tiles_y)/2

		tile_manhattan_error += current_tile_error
		count = count+1
		# out_data_X.append([frame_nos[frames[k]], x_act, x_pred, metric_X.get()])
		# out_data_Y.append([frame_nos[frames[k]], y_act, y_pred, metric_Y.get()])
		if current_tile_error != 0:
			total_error1+=1

		if actual_tile_row != pred_tile_row or actual_tile_col != pred_tile_col:
			total_error2+=1 

	return metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles,data_inp,data_act,total_error1,total_error2


def build_model(data, frame_nos, max_frame, tot_objects):
	model = linear_model.PARegressor(C=0.01, mode=2, eps=0.001, data=data, learning_rate=0.007, rho=0.99)
	metric_X = metrics.MAE()
	metric_Y = metrics.MAE()
	manhattan_error = []
	x_mae = []
	y_mae = []
	count=0
	total_error1=0
	total_error2=0
	i=0
	tile_manhattan_error=0
	act_tiles, pred_tiles = [],[]
	chunk_frames = []

	#Initial training of first 5 seconds
	prev_frames = {0}
	while True:
		curr_frame=frame_nos[i]
		prev_frames.add(i)
		if curr_frame<5*fps:
			i=i+1
			[inp_i,x,y]=data[curr_frame]
			model = model.fit_one(inp_i,x,y)
		else:
			break
	prev_frames = sorted(prev_frames)

	# Predicting frames and update model
	while True:
		curr_frame = frame_nos[i]
		nframe = min(pred_nframe, max_frame - frame_nos[i])

		if(nframe < 1):
			break

		frames = {i}
		for k in range(i+1, len(frame_nos)):
			if(frame_nos[k] < curr_frame + nframe):
				frames.add(k)
			else:
				i=k
				break
		
		if(i!=k):
			i=k

		if(i==(len(frame_nos)-1)):
			break
		frames = sorted(frames)
		chunk_frames.append(frames)

		metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles,data_inp,data_act,total_error1,total_error2 = pred_frames(data, model, metric_X, metric_Y, frames, prev_frames, tile_manhattan_error, act_tiles, pred_tiles, count,total_error1,total_error2)
		model = model.fit_n(frames)
		prev_frames = prev_frames+frames
		nn_model.fit(data_inp,data_act)
		manhattan_error.append(tile_manhattan_error*1.0 / count)
		x_mae.append(metric_X.get())
		y_mae.append(metric_Y.get())

		# print("Manhattan Tile Error: "+str(tile_manhattan_error*1.0 / count))
		# print(metric_X, metric_Y)
		# print("\n")

	return act_tiles, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae,total_error1,total_error2,count


def alloc_bitrate(pred_tiles, frame_nos, chunk_frames, pref_quality):
	vid_bitrate = []

	for i in range(len(chunk_frames)):
		chunk = chunk_frames[i]
		chunk_pred = pred_tiles[chunk[0]-chunk_frames[0][0] : chunk[-1]-chunk_frames[0][0]]
		chunk_bitrate = [[-1 for x in range(ncol_tiles)] for y in range(nrow_tiles)]
		chunk_weight = [[1. for x in range(ncol_tiles)] for y in range(nrow_tiles)]


		for tile in chunk_pred:
			tile_0 = nrow_tiles-1 if(tile[0]>=nrow_tiles) else tile[0]
			tile_1 = ncol_tiles-1 if(tile[1]>=ncol_tiles) else tile[1]

			tile_0 = 0 if(tile_0 < 0) else tile_0
			tile_1 = 0 if(tile_1 < 0) else tile_1

			chunk_weight[tile_0][tile_1] += 1.

			for j in range(nrow_tiles):
				for k in range(ncol_tiles):
					if not (j==tile_0 and k==tile_1):
						op1 = abs(tile_0-j) + abs(tile_1-k)
						op2, op3 = 0, 0
						op4 = 0
						if(j>tile_0):
							op2 = abs(tile_0-j+nrow_tiles) + abs(tile_1-k)
							op4 += abs(tile_0-j+nrow_tiles)
						else:
							op2 = abs(tile_0-j-nrow_tiles) + abs(tile_1-k)
							op4 += abs(tile_0-j-nrow_tiles)
						if(k>tile_1):
							op3 = abs(tile_0-j) + abs(tile_1-k+ncol_tiles)
							op4 += abs(tile_1-k+ncol_tiles)
						else:
							op3 = abs(tile_0-j) + abs(tile_1-k-ncol_tiles)
							op4 += abs(tile_1-k-ncol_tiles)

						dist = min(op1, op2, op3, op4)
						chunk_weight[j][k] += 1. - (1.0*dist)/((ncol_tiles+nrow_tiles)/2)

		total_weight = sum(sum(x) for x in chunk_weight)
		
		# print("Chunk Weight")
		# print(chunk_weight)

		for x in range(nrow_tiles):
			for y in range(ncol_tiles):
				chunk_bitrate[x][y] = chunk_weight[x][y]*pref_bitrate/total_weight;
		# print("Chunk Bitrate")
		# print(chunk_bitrate)
		# print(sum(sum(x) for x in chunk_bitrate))
		# print("\n")
		vid_bitrate.append(chunk_bitrate)

	return vid_bitrate


def calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames):
	qoe = 0
	prev_qoe_1 = 0
	weight_1 = 1
	weight_2 = 1
	weight_3 = 1
	tot1,tot2,tot3,tot4=0,0,0,0
	
	tile_width = width/ncol_tiles
	tile_height = height/nrow_tiles

	for i in range(len(chunk_frames)):
		qoe_1, qoe_2, qoe_3, qoe_4 = 0, 0, 0, 0
		tile_count = 0
		rows, cols = set(), set()
		rate = []

		chunk = chunk_frames[i]
		chunk_bitrate = vid_bitrate[i]
		chunk_act = act_tiles[chunk[0]-chunk_frames[0][0] : chunk[-1]-chunk_frames[0][0]]

		for j in range(len(chunk_act)):
			if(chunk_act[j][0] not in rows or chunk_act[j][1] not in cols):
				tile_count += 1
			rows.add(chunk_act[j][0])
			cols.add(chunk_act[j][1])
			row, col = chunk_act[j][0], chunk_act[j][1]

			# Find the number of tiles that can be accomodated from the center of the viewport
			n_tiles_width = math.ceil((player_width/2 - tile_width/2)/tile_width)
			n_tiles_height = math.ceil((player_height/2 - tile_height/2)/tile_height)
			tot_tiles = (2*n_tiles_width+1)*(2*n_tiles_height+1)

			local_qoe = 0
			local_rate = []  # a new metric to get the standard deviation of bitrate within the player view
			for x in range(2*n_tiles_height+1):
				for y in range(2*n_tiles_width+1):
					sub_row = row - n_tiles_height + x
					sub_col = col - n_tiles_width + y

					sub_row = nrow_tiles+row+sub_row if sub_row < 0 else sub_row
					sub_col = ncol_tiles+col+sub_col if sub_col < 0 else sub_col
					sub_row = sub_row-nrow_tiles if sub_row >= nrow_tiles else sub_row
					sub_col = sub_col-ncol_tiles if sub_col >= ncol_tiles else sub_col

					local_qoe += chunk_bitrate[sub_row][sub_col]
					local_rate.append(chunk_bitrate[sub_row][sub_col])

			qoe_1 += local_qoe / tot_tiles
			if(len(local_rate)>0):
				qoe_4 += np.std(local_rate)

			rate.append(local_qoe / tot_tiles)

		tile_count = 1 if tile_count==0 else tile_count
		qoe_1 /= tile_count
		qoe_4 /= tile_count

		if(len(rate)>0):
			qoe_2 = np.std(rate)
		qoe_2 /= tile_count

		if(i>0):
			qoe_3 = abs(prev_qoe_1 - qoe_1)

		qoe += qoe_1 - weight_1*qoe_2 - weight_2*qoe_3 - weight_3*qoe_4
		prev_qoe_1 = qoe_1

		tot1+=qoe_1
		tot2+=qoe_2
		tot3+=qoe_3
		tot4+=qoe_4

	return qoe,tot1,tot2,tot3,tot4

def main():
	data, frame_nos = [],[]
	data, frame_nos, max_frame, tot_objects = get_data(data, frame_nos, dataset, topic, usernum)

	act_tiles,pred_tiles,chunk_frames, manhattan_error, x_mae, y_mae = build_model(data, frame_nos, max_frame, tot_objects)

	i = 0
	while True:
		curr_frame=frame_nos[i]
		if curr_frame<5*fps:
			i += 1
		else:
			break

	frame_nos = frame_nos[i:]
	vid_bitrate = alloc_bitrate(pred_tiles, frame_nos, chunk_frames, pref_quality)
	qoe = calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames)

	print(qoe)

if __name__ == '__main__':
	main()