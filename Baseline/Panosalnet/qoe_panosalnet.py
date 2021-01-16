from sklearn import datasets
import numpy as np 
import math
import sys
import pickle
import matplotlib.pyplot as plt



dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
pref_quality = sys.argv[4]

nusers=0
matrix_error, manhattan_error, x_mae, y_mae, final_qoe = [],[],[],[],[]
save = './Predicted/QoE_Graphs/ds{}/'.format(dataset)
path_act = 'Viewport/ds{}/'.format(dataset)
path_pred = './head_prediction/ds{}/'.format(dataset)

player_width = 600
player_height = 300

if dataset == 1:
	nusers=58
	width=3840.0
	height=1920.0
	view_width = 3840.0
	view_height = 2048.0
	milisec = 1.0
elif dataset == 2:
	nusers=48
	width=2560.0
	height=1280.0
	view_width = 2560.0
	view_height = 1440.0
	milisec = 1.0

ncol_tiles = 16
nrow_tiles = 9

player_tiles_x = math.ceil(player_width*ncol_tiles*1.0/width)
player_tiles_y = math.ceil(player_height*nrow_tiles*1.0/height)

bitrates = {'360p':1, '480p':2.5, '720p':5, '1080p':8, '1440p':16}
pref_bitrate = bitrates[pref_quality]



def get_act_tiles(view_info, frame_nos):
	"""
	Calculate the tiles corresponding to the viewport
	"""
	act_viewport = []
	max_frame = int(view_info[-1][0]*1.0*fps/milisec)

	a = []
	b= []
	for i in range(len(view_info)-1):
		frame = int(view_info[i][0]*1.0*fps/milisec)
		frame_nos.append(frame)
		if(frame > max_frame):
			break

		view_x = int(view_info[i][1][1]*width/view_width)
		view_y = int(view_info[i][1][0]*height/view_height)
		tile_col = int(view_x * ncol_tiles / width)
		tile_row = int(view_y * nrow_tiles / height)

		act_viewport.append((tile_row, tile_col))

	return act_viewport, frame_nos, max_frame



def get_chunks(act_viewport, pred_viewport, frame_nos, max_frame):
	"""
	For chunks of fps number of frames for actual as well as predicted viewports
	"""
	act_tiles,pred_tiles,chunk_frames = [],[],[]
	chunk_size = fps
	number_of_chunks = int(len(act_viewport) / chunk_size)

	for i in range(number_of_chunks):
		act_tiles.append(act_viewport[i*chunk_size : (i+1)*chunk_size])
		pred_tiles.append(pred_viewport[i*chunk_size : (i+1)*chunk_size])
		chunk_frames.append(frame_nos[i*chunk_size : (i+1)*chunk_size])

	act_tiles.append(act_viewport[number_of_chunks*chunk_size :])
	pred_tiles.append(pred_viewport[number_of_chunks*chunk_size :])
	chunk_frames.append(frame_nos[number_of_chunks*chunk_size :])

	return act_tiles, pred_tiles, chunk_frames


def calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames):
	qoe = 0
	prev_qoe_1 = 0
	weight_1 = 1
	weight_2 = 1
	weight_3 = 1

	# PLayer viewport size
	player_width = 600
	player_height = 300
	tile_width = width/ncol_tiles
	tile_height = height/nrow_tiles

	for i in range(len(chunk_frames)):
		qoe_1, qoe_2, qoe_3, qoe_4 = 0, 0, 0, 0
		tile_count = 0
		rows, cols = set(), set()
		rate = []

		chunk = chunk_frames[i]
		chunk_bitrate = vid_bitrate[i]
		chunk_act = act_tiles[i]

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
				qoe_2 += np.std(local_rate)

			rate.append(local_qoe / tot_tiles)

		tile_count = 1 if tile_count==0 else tile_count
		qoe_1 /= tile_count
		qoe_2 /= tile_count

		if(len(rate)>0):
			qoe_3 = np.std(rate)
		qoe_3 /= tile_count

		if(i>0):
			qoe_4 = abs(prev_qoe_1 - qoe_1)

		qoe += qoe_1 - weight_1*qoe_2 - weight_2*qoe_3 - weight_3*qoe_4
		prev_qoe_1 = qoe_1

	return qoe


total_error1, total_error2, count_frames = 0,0,0

for usernum in range(nusers):
	print('User_{}'.format(usernum))
	user_manhattan_error = 0.


	viewport = pickle.load(open(path_act+"viewport_ds{}_topic{}_user{}".format(dataset, topic, usernum+1), "rb"), encoding='latin1')
	p_viewport = pickle.load(open(path_pred+"topic{}_user{}".format(topic, usernum), "rb"), encoding="latin1")


	frame_nos = []
	act_viewport, frame_nos, max_frame = get_act_tiles(viewport, frame_nos)

	# Predicted Tile = max of the probabilities in output
	pred_max_viewport = []
	for fr in range(len(p_viewport)):
		prob = p_viewport[fr]
		argmax = np.where(prob==prob.max())
		pred_max_viewport.append((argmax[0][0], argmax[1][0]))


	# Assert len(actual frames) = len(predicted frames)
	pred_viewport = p_viewport
	act_viewport = act_viewport[:len(pred_viewport)]
	frame_nos = frame_nos[:len(pred_viewport)]

	pred_viewport = pred_viewport[:len(act_viewport)]
	frame_nos = frame_nos[:len(pred_viewport)]



	# Calculate Manhattan Error
	for fr in range(len(pred_max_viewport)):
		act_tile = act_viewport[fr]
		pred_tile = pred_max_viewport[fr]

		# Get corrected error
		tile_col_dif = ncol_tiles
		tile_row_dif = act_tile[0] - pred_tile[0]
		tile_col_dif = min(pred_tile[1]-act_tile[1], act_tile[1]+ncol_tiles-pred_tile[1]) if act_tile[1] < pred_tile[1] else min(act_tile[1]-pred_tile[1], ncol_tiles+pred_tile[1]-act_tile[1])

		current_tile_error = abs(tile_row_dif) + abs(tile_col_dif)
		if(current_tile_error <= (player_tiles_x+player_tiles_y)/2):
			current_tile_error = 0
		else:
			current_tile_error -= (player_tiles_x+player_tiles_y)/2
		user_manhattan_error += current_tile_error

		if current_tile_error != 0:
			total_error1+=1

		if act_tile[0] != pred_tile[0] or act_tile[1] != pred_tile[1]:
			total_error2+=1

	manhattan_error.append(user_manhattan_error/len(pred_max_viewport))
	count_frames += len(act_viewport)


	act_tiles, pred_tiles, chunk_frames = get_chunks(act_viewport, pred_viewport, frame_nos, max_frame)


	# Allocate bitrate
	vid_bitrate = []
	for i in range(len(chunk_frames)):
		chunk = chunk_frames[i]
		chunk_pred = pred_tiles[i]
		chunk_bitrate = [[-1 for x in range(ncol_tiles)] for y in range(nrow_tiles)]
		chunk_weight = np.array([[0. for x in range(ncol_tiles)] for y in range(nrow_tiles)])

		for fr in chunk_pred:
			chunk_weight += fr

		total_weight = sum(sum(x) for x in chunk_weight)

		for x in range(nrow_tiles):
			for y in range(ncol_tiles):
				chunk_bitrate[x][y] = chunk_weight[x][y]*pref_bitrate/total_weight;

		vid_bitrate.append(chunk_bitrate)

	# Calculate QoE
	q = calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames)
	final_qoe.append(q)


avg_qoe = np.mean(final_qoe)
avg_manhattan_error = np.mean(manhattan_error)

print('Topic: '+topic)
print('PanoSalNet')
print('Pred_nframe: {}'.format(fps))
print('Avg. QoE: {}'.format(avg_qoe))
print('Avg. Manhattan error: {}'.format(avg_manhattan_error))
print('Total Error 1: {}'.format(total_error1))
print('Total Error 2: {}'.format(total_error2))
print('Count: {}'.format(count_frames))
print('\n\n')



