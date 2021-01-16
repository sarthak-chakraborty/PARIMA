from creme import linear_model
from creme import compose
from creme import compat
from creme import metrics
from creme import model_selection
from creme import optim
from creme import preprocessing
from creme import stream
from sklearn import datasets
import numpy as np 
import math
import sys
import pickle
from parima import build_model
from bitrate import alloc_bitrate
from qoe import calc_qoe

# fps fraction is the fraction of fps to be taken as chunk size (essentially the chunk size in seconds) Keep 1 as default
dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
offset = int(sys.argv[4])
pref_quality = sys.argv[5]
fps_fraction = float(sys.argv[6])

nusers=0
manhattan_error, x_mae, y_mae, final_qoe,qoe1,qoe2,qoe3,qoe4 = [],[],[],[],[],[],[],[]
matrix_error = []
tot_1,tot_2 = [],[]
avg_1,avg_2 = [],[]
count_frame = []
save = './Predicted/QoE_Graphs/ds{}/'.format(dataset)

if dataset == 1:
	nusers=58
elif dataset == 2:
	nusers=48
else:
	nusers=4

nrow_tiles = 8
ncol_tiles = 8

#Get viewport data
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

# Run for all users
for usernum in range(nusers):
	if usernum ==23: 
		continue
	print('User_{}'.format(usernum))
	user_matrix_error = 0.

	data, frame_nos = [],[]
	# Get data
	data, frame_nos, max_frame,total_objects = get_data(data, frame_nos, dataset, topic, usernum+1)
	# Build model
	act_tiles,pred_tiles,chunk_frames, err, x, y, total_error1, total_error2, count = build_model(data, frame_nos, max_frame,total_objects)
	
	if(len(pred_tiles)==0):
		continue

	manhattan_error.append(err)
	x_mae.append(x)
	y_mae.append(y)
	tot_1.append(total_error1)
	tot_2.append(total_error2)
	count_frame.append(count)

	i = 0
	while True:
		curr_frame=frame_nos[i]
		if curr_frame<5*fps:
			i += 1
		else:
			break

	# Find matrix error
	for fr in range(len(pred_tiles)):
		act_tile = act_tiles[fr]
		pred_tile = pred_tiles[fr]
		act_prob = np.array([[0. for i in range(ncol_tiles)] for j in range(nrow_tiles)])
		pred_prob = np.array([[0. for i in range(ncol_tiles)] for j in range(nrow_tiles)])
		act_prob[act_tile[0]%nrow_tiles][act_tile[1]%ncol_tiles] = 1

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

	matrix_error.append(user_matrix_error/len(pred_tiles))


	frame_nos = frame_nos[i:]
	# Allocate bitrates and calculate QoE
	vid_bitrate = alloc_bitrate(pred_tiles, frame_nos, chunk_frames, pref_quality)
	q,q1,q2,q3,q4 = calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames)
	final_qoe.append(q)
	qoe1.append(q1)
	qoe2.append(q2)
	qoe3.append(q3)
	qoe4.append(q4)
	print(q,err[-1])

#Find averaged results
avg_qoe = np.mean(final_qoe)
avg_qoe1 = np.mean(qoe1)
avg_qoe2 = np.mean(qoe2)
avg_qoe3 = np.mean(qoe3)
avg_qoe4 = np.mean(qoe4)
avg_manhattan_error = np.mean(list(zip(*manhattan_error)), axis=1)
avg_matrix_error = np.mean(matrix_error)
avg_x_mae = np.mean(list(zip(*x_mae)), axis=1)
avg_y_mae = np.mean(list(zip(*y_mae)), axis=1)

#Print averaged results
print('Topic: '+topic)
print('Qoe PA TS Var1 16 9')
print('Pred nframe',(fps*fps_fraction))
print('Avg. QoE: ',avg_qoe,avg_qoe1,avg_qoe2,avg_qoe3,avg_qoe4)
print('Avg. Manhattan Error: {}'.format(avg_manhattan_error[-1]))
print('Avg. Matrix Error: {}'.format(avg_matrix_error))
print('Avg. X_MAE: {}'.format(avg_x_mae[-1]))
print('Avg. Y_MAE: {}'.format(avg_y_mae[-1]))
print('Total Error 1:',sum(tot_1))
print('Total Error 2:',sum(tot_2))
print('Count',sum(count_frame))

print('\n\n')

