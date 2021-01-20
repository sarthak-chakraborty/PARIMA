import numpy as np 
import math
import pickle


def alloc_bitrate(pred_tiles, chunk_frames, nrow_tiles, ncol_tiles, pref_bitrate, player_tiles_x, player_tiles_y):
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
						if(dist<=(player_tiles_x+player_tiles_y)/2):
							chunk_weight[j][k] += 1. - (1.0*dist/2)/((ncol_tiles+nrow_tiles)/2)
						else:
							chunk_weight[j][k] += 1. - (1.0*dist)/((ncol_tiles+nrow_tiles)/2)

		total_weight = sum(sum(x) for x in chunk_weight)

		for x in range(nrow_tiles):
			for y in range(ncol_tiles):
				chunk_bitrate[x][y] = chunk_weight[x][y]*pref_bitrate/total_weight;

		vid_bitrate.append(chunk_bitrate)
	
	return vid_bitrate
