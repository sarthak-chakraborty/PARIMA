import numpy as np 
import math


def calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames, width, height, nrow_tiles, ncol_tiles, player_width, player_height):
	qoe = 0
	prev_qoe_1 = 0
	weight_1 = 1
	weight_2 = 1
	weight_3 = 1

	# PLayer viewport size
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