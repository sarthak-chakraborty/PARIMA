from creme import linear_model
from creme import compose
from creme import compat
from creme import metrics
from creme import model_selection
from creme import optim
from creme import preprocessing
from creme import stream
from sklearn import datasets
import numpy as np 
import math
import sys
import pickle

dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
pref_quality = sys.argv[4]

usernum=1
ncol_tiles=8
nrow_tiles=8
pred_nframe=1
bitrates = {'360p':1, '480p':2.5, '720p':5, '1080p':8, '1440p':16}	# [360p, 480p, 720p, 1080p, 1440p]

pref_bitrate = bitrates[pref_quality]

# # OUR DATA
# width=3840.0
# height=1920.0
# view_width = 1280
# view_height = 720
# milisec = 1000.0

# # ds 1
# width=3840.0
# height=1920.0
# view_width = 3840.0
# view_height = 2048.0
# milisec = 1.0

# ds 2
width=2560.0
height=1280.0
view_width = 2560.0
view_height = 1440.0
milisec = 1.0


def get_data(data, frame_nos, dataset, topic, usernum):

	# # FOR OWN DATA
	# obj_info = np.load('lovish_stitched_object_trajectory_converted.npy', allow_pickle=True).item()
	# view_info = np.load('Viewport'+str(usernum)+'.npy', allow_pickle=True, encoding='latin1')

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
	
	return data, frame_nos, max_frame



def pred_frames(data, model, metric_X, metric_Y, frames, tile_manhattan_error, act_tiles, pred_tiles):
	x_pred, y_pred = 0,0
	count = 0

	for k in range(len(frames)):
		[inp_k, x_act, y_act] = data[frames[k]]
		if(k == 0):
			x_pred, y_pred = model.predict_one(inp_k)
		else:
			inp_k['VIEWPORT_x'] = x_pred
			inp_k['VIEWPORT_y'] = y_pred
			x_pred, y_pred = model.predict_one(inp_k)	

		metric_X = metric_X.update(x_act, x_pred)
		metric_Y = metric_Y.update(y_act, y_pred)

		actual_tile_col = int(x_act * ncol_tiles / width)
		actual_tile_row = int(y_act * nrow_tiles / height)
		pred_tile_col = int(x_pred * ncol_tiles / width)
		pred_tile_row = int(y_pred * nrow_tiles / height)

		actual_tile_row = nrow_tiles-1 if(actual_tile_row >= nrow_tiles) else actual_tile_row
		actual_tile_col = ncol_tiles-1 if(actual_tile_col >= ncol_tiles) else actual_tile_col


		####################################################
		print("x: "+str(x_act))
		print("x_pred: "+str(x_pred))
		print("y: "+str(y_act))	
		print("y_pred: "+str(y_pred))
		print("("+str(actual_tile_row)+","+str(actual_tile_col)+"),("+str(pred_tile_row)+","+str(pred_tile_col)+")")
		#######################################################


		act_tiles.append((actual_tile_row, actual_tile_col))
		pred_tiles.append((pred_tile_row, pred_tile_col))

		tile_manhattan_error += abs(actual_tile_row - pred_tile_row) + abs(actual_tile_col - pred_tile_col)
		count = count+1
		# out_data_X.append([frame_nos[frames[k]], x_act, x_pred, metric_X.get()])
		# out_data_Y.append([frame_nos[frames[k]], y_act, y_pred, metric_Y.get()])

	return metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles


def build_model(data, frame_nos, max_frame):
	model = linear_model.PARegressor(C=0.01, mode=2, eps=0.1, data=data)
	metric_X = metrics.MAE()
	metric_Y = metrics.MAE()
	manhattan_error = []
	x_mae = []
	y_mae = []

	i=0
	tile_manhattan_error=0
	act_tiles, pred_tiles = [],[]
	chunk_frames = []

	#Initial training of first 5 seconds
	while True:
		curr_frame=frame_nos[i]
		if curr_frame<5*fps:
			i=i+1
			[inp_i,x,y]=data[curr_frame]
			model = model.fit_one(inp_i,x,y)
		else:
			break

	# Predicting frames and update model
	while True:
		curr_frame = frame_nos[i]
		nframe = min(pred_nframe, max_frame - frame_nos[i])

		if(nframe <= 0):
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

		print(len(frames))
		metric_X, metric_Y, tile_manhattan_error, count, act_tiles, pred_tiles = pred_frames(data, model, metric_X, metric_Y, frames, tile_manhattan_error, act_tiles, pred_tiles)
		model = model.fit_n(frames)

		manhattan_error.append(tile_manhattan_error*1.0 / count)
		x_mae.append(metric_X.get())
		y_mae.append(metric_Y.get())

		print("Manhattan Tile Error: "+str(tile_manhattan_error*1.0 / count))
		print(metric_X, metric_Y)
		print("\n")

	return act_tiles, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae


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

	for i in range(len(chunk_frames)):
		qoe_1, qoe_3 = 0, 0
		tile_count = 0
		row, col = -1, -1
		rate = []

		chunk = chunk_frames[i]
		chunk_bitrate = vid_bitrate[i]
		chunk_act = act_tiles[chunk[0]-chunk_frames[0][0] : chunk[-1]-chunk_frames[0][0]+1]


		for j in range(len(chunk_act)):
			if(chunk_act[j][0]!=row or chunk_act[j][1]!=col):
				tile_count += 1
			row, col = chunk_act[j][0], chunk_act[j][1]
			qoe_1 += chunk_bitrate[row][col]
			rate.append(chunk_bitrate[row][col])


		qoe_1 /= tile_count

		qoe_2 = np.std(rate)
		qoe_2 /= tile_count

		if(i>0):
			qoe_3 = abs(prev_qoe_1 - qoe_1)

		qoe += qoe_1 - weight_1*qoe_2 - weight_2*qoe_3
		prev_qoe_1 = qoe_1

	return qoe



def main():
	data, frame_nos = [],[]
	data, frame_nos, max_frame = get_data(data, frame_nos, dataset, topic, usernum)

	act_tiles,pred_tiles,chunk_frames, manhattan_error, x_mae, y_mae = build_model(data, frame_nos, max_frame)

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