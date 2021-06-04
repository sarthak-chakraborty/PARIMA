import numpy as np 
import math
import pickle
from naba import get_data, tiling, alloc_bitrate, calc_qoe
import argparse
import json


def main():

	parser = argparse.ArgumentParser(description='Run NABA algorithm and calculate Average QoE of a video for all users')

	parser.add_argument('-D', '--dataset', type=int, required=True, help='Dataset ID (1 or 2)')
	parser.add_argument('-T', '--topic', required=True, help='Topic in the particular Dataset (video name)')
	parser.add_argument('--fps', type=int, required=True, help='fps of the video')
	parser.add_argument('-O', '--offset', type=int, default=0, help='Offset for the start of the video in seconds (when the data was logged in the dataset) [default: 0]')
	parser.add_argument('--fpsfrac', type=float, default=1.0, help='Fraction with which fps is to be multiplied to change the chunk size [default: 1.0]')
	parser.add_argument('-Q', '--quality', required=True, help='Preferred bitrate quality of the video (360p, 480p, 720p, 1080p, 1440p)')

	args = parser.parse_args()

	if args.dataset != 1 and args.dataset != 2:
		print("Incorrect value of the Dataset ID provided!!...")
		print("======= EXIT ===========")
		exit()

	pred_nframe = args.fps * args.fpsfrac

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

	final_qoe = []

	for usernum in range(nusers):
		print('User_{}'.format(usernum))

		data, frame_nos = [],[]
		data, frame_nos, max_frame = get_data(data, frame_nos, args.dataset, args.topic, usernum+1, args.fps, milisec, width, height, view_width, view_height)

		act_tiles, chunk_frames = tiling(data, frame_nos, max_frame, width, height, nrow_tiles, ncol_tiles, args.fps, pred_nframe)

		# To be consistent with our model
		i = 0
		while True:
			curr_frame=frame_nos[i]
			if curr_frame<5*args.fps:
				i += 1
			else:
				break

		frame_nos = frame_nos[i:]
		vid_bitrate = alloc_bitrate(frame_nos, chunk_frames, pref_bitrate, nrow_tiles, ncol_tiles)
		q = calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames, width, height, nrow_tiles, ncol_tiles, player_width, player_height)
		final_qoe.append(q)
		print("QoE: {}".format(q))

	# Find averaged results
	final_qoe.sort()
	avg_qoe = np.mean(final_qoe)

	# Print averaged results
	print('Topic: '+topic)
	print('Qoe NABA')
	print('Pred nframe',(args.fps*args.fpsfrac))
	print('Avg. QoE: ',avg_qoe)

	print('\n\n')

if __name__ == "__main__":
	main()
