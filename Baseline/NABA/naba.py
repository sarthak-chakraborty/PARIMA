import numpy as np 
import math
import pickle


def get_data(data, frame_nos, dataset, topic, usernum, fps, milisec, width, height, view_width, view_height):
	"""
	Read and return the viewport data
	"""
	VIEW_PATH = '../../Viewport/'

	view_info = pickle.load(open(VIEW_PATH + 'ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, usernum), 'rb'), encoding='latin1')

	if dataset == 1:
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
			data.append((X, int(view_info[i+1][1][1]*width/view_width),int(view_info[i+1][1][0]*height/view_height)))

	elif dataset == 2:
		for k in range(len(view_info)-1):
			if view_info[k][0]<=offset+60 and view_info[k+1][0]>offset+60:
				max_frame = int(view_info[k][0]*1.0*fps/milisec)
				break
		
		for k in range(len(view_info)-1):
			if view_info[k][0]<=offset and view_info[k+1][0]>offset:
				min_index = k+1
				break	

		prev_frame = 0
		for i in range(min_index,len(view_info)-1):
			frame = int((view_info[i][0])*1.0*fps/milisec)

			if frame == prev_frame:
				continue
			
			if(frame > max_frame):
				break

			frame_nos.append(frame)
			
			X={}
			X['VIEWPORT_x']=int(view_info[i][1][1]*width/view_width)
			X['VIEWPORT_y']=int(view_info[i][1][0]*height/view_height)
			data.append((X, int(view_info[i+1][1][1]*width/view_width),int(view_info[i+1][1][0]*height/view_height)))
			prev_frame = frame
			
	return data, frame_nos, max_frame


def tiling(data, frame_nos, max_frame, width, height, nrow_tiles, ncol_tiles, fps, pred_nframe):
	"""
	Calculate the tiles corresponding to the viewport and segment them into different chunks
	"""
	count=0
	i=0
	act_tiles = []
	chunk_frames = []

	# Leaving the first 5 seconds ( to keep consistent with our model)
	while True:
		curr_frame = frame_nos[i]
		if curr_frame<5*fps:
			i=i+1
			[inp_i,x,y]=data[curr_frame]
		else:
			break


	# Calulate the tiles and store it in chunks
	while True:
		curr_frame = frame_nos[i]
		nframe = min(pred_nframe, max_frame - frame_nos[i])
		if(nframe <= 0):
			break
		
		# Add the frames that will be in the current chunk
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

		# Get the actual tile
		for k in range(len(frames)):
			[inp_k, x_act, y_act] = data[frames[k]]
			# print(x_act, y_act)

			actual_tile_col = int(x_act * ncol_tiles / width)
			actual_tile_row = int(y_act * nrow_tiles / height)
			# print(actual_tile_col, actual_tile_row)

			actual_tile_row = actual_tile_row-nrow_tiles if(actual_tile_row >= nrow_tiles) else actual_tile_row
			actual_tile_col = actual_tile_col-ncol_tiles if(actual_tile_col >= ncol_tiles) else actual_tile_col
			actual_tile_row = actual_tile_row+nrow_tiles if actual_tile_row < 0 else actual_tile_row
			actual_tile_col = actual_tile_col+ncol_tiles if actual_tile_col < 0 else actual_tile_col
			# print(actual_tile_col, actual_tile_row)
			# print()
			act_tiles.append((actual_tile_row, actual_tile_col))

	return act_tiles, chunk_frames



def alloc_bitrate(frame_nos, chunk_frames, pref_bitrate, nrow_tiles, ncol_tiles):
	"""
	Allocates equal bitrate to all the tiles
	"""
	vid_bitrate = []

	for i in range(len(chunk_frames)):
		chunk = chunk_frames[i]
		chunk_bitrate = [[-1 for x in range(ncol_tiles)] for y in range(nrow_tiles)]
		chunk_weight = [[1. for x in range(ncol_tiles)] for y in range(nrow_tiles)]

		total_weight = sum(sum(x) for x in chunk_weight)

		for x in range(nrow_tiles):
			for y in range(ncol_tiles):
				chunk_bitrate[x][y] = chunk_weight[x][y]*pref_bitrate/total_weight;

		vid_bitrate.append(chunk_bitrate)

	return vid_bitrate



def calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames, width, height, nrow_tiles, ncol_tiles, player_width, player_height):
	"""
	Calculate QoE based on the video bitrates
	"""
	qoe = 0
	prev_qoe_1 = 0
	weight_1 = 1
	weight_2 = 1
	weight_3 = 1
	
	tile_width = width/ncol_tiles
	tile_height = height/nrow_tiles


	for i in range(len(chunk_frames[:55])):
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
			tot_tiles = (2 * n_tiles_width+1) * (2 * n_tiles_height+1)

			local_qoe = 0
			local_rate = []  # a new metric to get the standard deviation of bitrate within the player view (qoe2)
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

