# FIle to run to get QoE summarised results for Clust
import pickle
import math
import numpy as np
import sys

ds = int(sys.argv[1])
topic = sys.argv[2]

width_dict = {1: 3840, 2: 2560}
height_dict = {1: 2048, 2: 1440}

# ds 1
width = width_dict[ds]
height = height_dict[ds]
nrow_tiles = 8
ncol_tiles = 8

bitrates = {'360p':1, '480p':2.5, '720p':5, '1080p':8, '1440p':16}	# [360p, 480p, 720p, 1080p, 1440p]
player_width = 600
player_height = 300
pref_bitrate = bitrates['1080p']

milisec = 1.0
player_tiles_x = math.ceil(player_width*ncol_tiles*1.0/width)
player_tiles_y = math.ceil(player_height*nrow_tiles*1.0/height)



def min_manhattan(act, pred):
	"""
	Get minimum manhattan distance
	"""
	x_min = min(abs(pred[0]-act[0]), abs(pred[0]+ncol_tiles-act[0]), (abs(pred[0]-ncol_tiles-act[0])))
	y_min = min(abs(pred[1]-act[1]), abs(pred[1]+nrow_tiles-act[1]), (abs(pred[1]-nrow_tiles-act[1])))
	return x_min + y_min




def alloc_bitrate(pred_tiles, frame_nos, chunk_frames, pref_quality):
	"""
	Allocate bitrate based on pyramid scheme
	"""
	vid_bitrate = []

	for i in range(len(chunk_frames)):
		chunk = chunk_frames[i]
		chunk_pred = pred_tiles[chunk[0]-chunk_frames[0][0] : chunk[-1]-chunk_frames[0][0]]
		chunk_bitrate = [[-1 for x in range(ncol_tiles)] for y in range(nrow_tiles)]
		chunk_weight = [[1. for x in range(ncol_tiles)] for y in range(nrow_tiles)]

		for tile in chunk_pred:
			tile_0 =  tile[0] % nrow_tiles
			tile_1 = tile[1]%ncol_tiles

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
				chunk_bitrate[x][y] = chunk_weight[x][y]*pref_bitrate/total_weight
		vid_bitrate.append(chunk_bitrate)

	return vid_bitrate



# Calculate QoE of user
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
	# Load actual and predicted viewports
	act_viewport = pickle.load(open("viewportds" + str(ds) + "/act_viewport_ds"+ str(ds) + "_topic" + topic + "", "rb"))
	pred_viewport = pickle.load(open("viewportds" + str(ds) + "/pred_viewport_ds"+ str(ds) + "_topic" + topic + "", "rb"))

	pred_viewport_tile = pred_viewport.copy()
	act_viewport_tile = act_viewport.copy()

	for u in pred_viewport_tile.keys():
		try:
			pred_viewport_tile[u] = pred_viewport_tile[u][:55]                      # Considering 55 chunks for model comparison
			act_viewport_tile[u]  = act_viewport_tile[u][:55]
			act_viewport[u] = act_viewport[u][:55*len(act_viewport[u][0])]
			pred_viewport[u] = pred_viewport[u][:55*len(pred_viewport[u][0])]
		except:
			pass

	for u in pred_viewport_tile.keys():
		for i in range(len(pred_viewport_tile[u])):
			for f in range(len(pred_viewport_tile[u][i])):
				pred_viewport_tile[u][i][f] = (int(pred_viewport_tile[u][i][f][0]/width * 8)%8, int(pred_viewport_tile[u][i][f][1]/height * 8)%8)      # Convert viewport to tile

	for u in act_viewport_tile.keys():
		for i in range(len(act_viewport_tile[u])):
			for f in range(len(act_viewport_tile[u][i])):
				act_viewport_tile[u][i][f] = (int(act_viewport_tile[u][i][f][0]/width * 8)%8, int(act_viewport_tile[u][i][f][1]/height * 8)%8)          # Convert viewport to tile


	# Getting the lists into required format for alloc bitrate and qoe
	for u in pred_viewport_tile.keys():
		l=[]
		for x in pred_viewport_tile[u]:
			l = l + x
		pred_viewport_tile[u]=l                        

	for u in act_viewport_tile.keys():
		l=[]
		for x in act_viewport_tile[u]:
			l = l + x
		act_viewport_tile[u]=l


	total_qoe = 0
	total_manhattan = 0
	total_matrix_error = 0
	for u in pred_viewport_tile.keys():
		frame_nos = []
		chunk_frames = []
		pred_tiles = pred_viewport_tile[u]
		pred = pred_viewport[u]
		act_tiles = act_viewport_tile[u]
		f = 0
		for x in pred:
			frames = []
			for y in x:
				frames.append(f)
				frame_nos.append(f)
				f = f + 1
			chunk_frames.append(frames)                 # Generate chunks, find manhattan error

		count = 0
		manhattan = 0
		for i in range(len(pred_tiles)):
			manhattan = manhattan + min_manhattan(act_tiles[i], pred_tiles[i])
			count = count + 1
		total_manhattan = total_manhattan + manhattan*1.0/count

		vid_bitrate = alloc_bitrate	(pred_viewport_tile[u], frame_nos, chunk_frames, '1080p')                  # Get qoe
		qoe,qoe1,qoe2,qoe3,qoe4 = calc_qoe (vid_bitrate, act_viewport_tile[u], frame_nos, chunk_frames)
		total_qoe = total_qoe + qoe
		

	avg_qoe = total_qoe / len(pred_viewport.keys())
	avg_manhattan = total_manhattan / len(pred_viewport.keys())	
	avg_matrix_error = total_matrix_error / len(pred_viewport.keys())
	print('QoE: ', avg_qoe)
	print('Manhattanerror: ', avg_manhattan)



if __name__ == "__main__":
	main()



