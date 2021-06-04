import numpy as np 
import math
import sys
import pickle
import argparse
import json

from qoe import calc_qoe
from utils import get_act_tiles, get_chunks, alloc_bitrate


def main():

	parser = argparse.ArgumentParser(description='Calculate QoE and error for PanoSalNet algorithm')

	parser.add_argument('-D', '--dataset', type=int, required=True, help='Dataset ID (1 or 2)')
	parser.add_argument('-T', '--topic', required=True, help='Topic in the particular Dataset (video name)')
	parser.add_argument('--fps', type=int, required=True, help='fps of the video')
	parser.add_argument('-Q', '--quality', required=True, help='Preferred bitrate quality of the video (360p, 480p, 720p, 1080p, 1440p)')

	args = parser.parse_args()

	if args.dataset != 1 and args.dataset != 2:
		print("Incorrect value of the Dataset ID provided!!...")
		print("======= EXIT ===========")
		exit()

	# Get the necessary information regarding the dimensions of the video
	print("Reading JSON...")
	file = open('./meta.json', )
	jsonRead = json.load(file)

	nusers = jsonRead["dataset"][args.dataset-1]["nusers"]
	width = jsonRead["dataset"][args.dataset-1]["width"]
	height = jsonRead["dataset"][args.dataset-1]["height"]
	view_width = jsonRead["dataset"][args.dataset-1]["view_width"]
	view_height = jsonRead["dataset"][args.dataset-1]["view_height"]
	milisec = jsonRead["dataset"][args.dataset-1]["milisec"]

	pref_bitrate = jsonRead["bitrates"][args.quality]
	ncol_tiles = jsonRead["ncol_tiles"]
	nrow_tiles = jsonRead["nrow_tiles"]
	player_width = jsonRead["player_width"]
	player_height = jsonRead["player_height"]

	player_tiles_x = math.ceil(player_width*ncol_tiles*1.0/width)
	player_tiles_y = math.ceil(player_height*nrow_tiles*1.0/height)

	PATH_ACT = '../../Viewport/ds{}/'.format(args.dataset)
	PATH_PRED = './head_prediction/ds{}/'.format(args.dataset)

	manhattan_error, x_mae, y_mae, final_qoe = [],[],[],[]
	count_frames = 0
	for usernum in range(nusers):
		print('User_{}'.format(usernum))
		user_manhattan_error = 0.

		viewport = pickle.load(open(PATH_ACT + "viewport_ds{}_topic{}_user{}".format(dataset, topic, usernum+1), "rb"), encoding='latin1')
		p_viewport = pickle.load(open(PATH_PRED + "topic{}_user{}".format(topic, usernum), "rb"), encoding="latin1")

		frame_nos = []
		act_viewport, frame_nos, max_frame = get_act_tiles(viewport, frame_nos, args.fps, args.milisec, width, height, view_width, view_height)

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
			user_manhattan_error += current_tile_error


		manhattan_error.append(user_manhattan_error/len(pred_max_viewport))
		count_frames += len(act_viewport)


		act_tiles, pred_tiles, chunk_frames = get_chunks(act_viewport, pred_viewport, frame_nos, max_frame, args.fps)

		# Allocate bitrate
		vid_bitrate = alloc_bitrate(pred_tiles, chunk_frames, nrow_tiles, ncol_tiles, pref_bitrate)

		# Calculate QoE
		q = calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames, width, height, nrow_tiles, ncol_tiles, player_width, player_height)
		final_qoe.append(q)


	avg_qoe = np.mean(final_qoe)
	avg_manhattan_error = np.mean(manhattan_error)

	#Print averaged results
	print("\n======= RESULTS ============")
	print('PanoSalNet')
	print('Dataset: {}'.format(args.dataset))
	print('Topic: ' + args.topic)
	print('Pred_nframe: {}'.format(args.fps))
	print('Avg. QoE: {}'.format(avg_qoe))
	print('Avg. Manhattan error: {}'.format(avg_manhattan_error))
	print('Count: {}'.format(count_frames))
	print('\n\n')


if __name__ == "__main__":
	main()
