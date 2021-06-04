from sklearn import datasets
import numpy as np 
import math
import pickle
import json
import argparse
from parima import build_model
from bitrate import alloc_bitrate
from qoe import calc_qoe



#Get viewport data
def get_data(data, frame_nos, dataset, topic, usernum, fps, milisec, width, height, view_width, view_height):
	VIEW_PATH = '../Viewport/'
	OBJ_PATH = '../Obj_traj/'

	obj_info = np.load(OBJ_PATH + 'ds{}/ds{}_topic{}.npy'.format(dataset, dataset, topic), allow_pickle=True,  encoding='latin1').item()
	view_info = pickle.load(open(VIEW_PATH + 'ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, usernum), 'rb'), encoding='latin1')


	n_objects = []
	for i in obj_info.keys():
		try:
			n_objects.append(max(obj_info[i].keys()))
		except:
			n_objects.append(0)
	total_objects=max(n_objects)

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
			prev_frame = frame
			
	return data, frame_nos, max_frame,total_objects


def calc_matrix_error(act_tiles, pred_tiles, ncol_tiles, nrow_tiles):
	user_matrix_error = 0.

	for fr in range(len(pred_tiles)):
		act_tile = act_tiles[fr]
		pred_tile = pred_tiles[fr]
		act_prob = np.array([[0. for i in range(ncol_tiles)] for j in range(nrow_tiles)])
		pred_prob = np.array([[0. for i in range(ncol_tiles)] for j in range(nrow_tiles)])
		act_prob[act_tile[0] % nrow_tiles][act_tile[1] % ncol_tiles] = 1

		x = nrow_tiles-1 if pred_tile[0] >= nrow_tiles else pred_tile[0]
		x = 0 if x < 0 else x
		y = ncol_tiles-1 if pred_tile[1] >= ncol_tiles else pred_tile[1] 
		y = 0 if y < 0 else y
		pred_prob[x][y] = 1

		d=0.
		for i in range(nrow_tiles):
			for j in range(ncol_tiles):
				d += np.square(pred_prob[i][j] - act_prob[i][j])
		user_matrix_error += np.sqrt(d)

	return user_matrix_error / len(pred_tiles)


def main():
	parser = argparse.ArgumentParser(description='Run PARIMA algorithm and calculate Average QoE of a video for all users')

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

	player_tiles_x = math.ceil(player_width*ncol_tiles*1.0/width)
	player_tiles_y = math.ceil(player_height*nrow_tiles*1.0/height)

	manhattan_error, x_mae, y_mae, qoe = [],[],[],[]
	matrix_error = []

	# Run for all users
	for usernum in range(nusers):
		print('User_{}'.format(usernum))

		data, frame_nos = [],[]
		# Get data
		data, frame_nos, max_frame,total_objects = get_data(data, frame_nos, args.dataset, args.topic, usernum+1, args.fps, milisec, width, height, view_width, view_height)
		# Build model
		act_tiles, pred_tiles, chunk_frames, err, x, y = build_model(data, frame_nos, max_frame, tot_objects, width, height, nrow_tiles, ncol_tiles, args.fps, pred_nframe)
		
		if(len(pred_tiles)==0):
			continue

		manhattan_error.append(err)
		x_mae.append(x)
		y_mae.append(y)

		i = 0
		while True:
			curr_frame = frame_nos[i]
			if curr_frame < 5 * args.fps:
				i += 1
			else:
				break
	
		mat_err = calc_matrix_error(act_tiles, pred_tiles, ncol_tiles, nrow_tiles)
		matrix_error.append(mat_err)

		frame_nos = frame_nos[i:]
		# Allocate bitrates and calculate QoE
		vid_bitrate = alloc_bitrate(pred_tiles, chunk_frames, nrow_tiles, ncol_tiles, pref_bitrate, player_tiles_x, player_tiles_y)
		q = calc_qoe(vid_bitrate, act_tiles, chunk_frames, width, height, nrow_tiles, ncol_tiles, player_width, player_height)
		qoe.append(q)
		print("QoE: {}".format(q))


	#Find averaged results
	avg_qoe = np.mean(qoe)
	avg_manhattan_error = np.mean(list(zip(*manhattan_error)), axis=1)
	avg_matrix_error = np.mean(matrix_error)
	avg_x_mae = np.mean(list(zip(*x_mae)), axis=1)
	avg_y_mae = np.mean(list(zip(*y_mae)), axis=1)

	#Print averaged results
	print("\n======= RESULTS ============")
	print('PARIMA')
	print('Dataset: {}'.format(args.dataset))
	print('Topic: {}'.format(args.topic))
	print('Pred nframe: {}'.format(args.fps * args.fpsfrac))
	print('Avg. QoE: {}'.format(avg_qoe))
	print('Avg. Manhattan Error: {}'.format(avg_manhattan_error[-1]))
	print('Avg. Matrix Error: {}'.format(avg_matrix_error))
	print('Avg. X_MAE: {}'.format(avg_x_mae[-1]))
	print('Avg. Y_MAE: {}'.format(avg_y_mae[-1]))

	print('\n\n')


if __name__ == '__main__':
	main()
