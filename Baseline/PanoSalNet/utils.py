import numpy as np 
import math
import pickle

def get_act_tiles(view_info, frame_nos, fps, milisec, width, height, view_width, view_height):
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



def get_chunks(act_viewport, pred_viewport, frame_nos, max_frame, fps):
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


def alloc_bitrate(pred_tiles, chunk_frames, nrow_tiles, ncol_tiles, pref_bitrate):
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