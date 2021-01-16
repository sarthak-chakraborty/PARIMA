import numpy as np 
import pandas as pd
import math
import sys
import pickle
import random
import warnings
import time
from parima import build_model
from bitrate import alloc_bitrate
from qoe import calc_qoe


dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
offset=int(sys.argv[4])
pref_quality = sys.argv[5]

usernum=2
ncol_tiles=8
nrow_tiles=8
pred_nframe=fps
bitrates = {'360p':1, '480p':2.5, '720p':5, '1080p':8, '1440p':16}	# [360p, 480p, 720p, 1080p, 1440p]

pref_bitrate = bitrates[pref_quality]

# ds 1
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

	obj_info = np.load('../../Obj_traj/ds{}/ds{}_topic{}.npy'.format(dataset, dataset, topic), allow_pickle=True,  encoding='latin1').item()
	view_info = pickle.load(open('../../Viewport/ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, usernum), 'rb'), encoding='latin1')


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



def main():
	data, frame_nos = [],[]
	print("Reading Viewport Data and Object Trajectories...")
	data, frame_nos, max_frame, tot_objects = get_data(data, frame_nos, dataset, topic, usernum)
	print("Data read\n")
	
	print("Build Model...")
	act_tiles, pred_tiles, chunk_frames, manhattan_error, x_mae, y_mae = build_model(data, frame_nos, max_frame, tot_objects, width, height, nrow_tiles, ncol_tiles, fps, pred_nframe)

	i = 0
	while True:
		curr_frame=frame_nos[i]
		if curr_frame<5*fps:
			i += 1
		else:
			break

	frame_nos = frame_nos[i:]
	print("Allocate Bitrates...")
	vid_bitrate = alloc_bitrate(pred_tiles, frame_nos, chunk_frames, pref_quality, nrow_tiles, ncol_tiles, pref_bitrate)
	
	print("Calculate QoE...")
	qoe = calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames, width, height, nrow_tiles, ncol_tiles)

	print(qoe)

if __name__ == '__main__':
	main()
