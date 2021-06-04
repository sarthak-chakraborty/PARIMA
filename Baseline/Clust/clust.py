import os
import numpy as np
import pickle
import header
import sys
import random
import numpy.matlib as npm
from Quaternion import Quat
import pickle
import Quaternion
import argparse

import head_orientation_lib
import saldat_head_orientation



# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
	# Number of quaternions to average
	M = Q.shape[0]
	A = npm.zeros(shape=(4,4))

	for i in range(0,M):
		q = Q[i,:]
		# multiply q with its transposed version q' and add A
		A = np.outer(q,q) + A

	# scale
	A = (1.0/M)*A
	# compute eigenvalues and -vectors
	eigenValues, eigenVectors = np.linalg.eig(A)
	# Sort by largest eigenvalue
	eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
	# return the real part of the largest eigenvector (has only real part)
	return np.real(eigenVectors[:,0].A1)



def interpolate(mean):
	if mean[0] is None:
		i = 1
		while mean[i] is None:
			i += 1
			if i >= len(mean):
				break
		if i < len(mean):
			mean[0] = mean[i]
		else:
			print("Exiting")
			exit(0)
	
	for k in range(1, len(mean)):
		if mean[k] is None:
			mean[k] = mean[k-1]
	return mean



if __name__ == "__main__":

	#specify dataset & video name to extract
	parser = argparse.ArgumentParser(description='Run Viewport Extraction Algorithm')

	parser.add_argument('-D', '--dataset', type=int, required=True, help='Dataset ID (1 or 2)')
	parser.add_argument('-T', '--topic', required=True, help='Topic in the particular Dataset (video name)')
	parser.add_argument('--fps', type=int, required=True, help='fps of the video')
	
	args = parser.parse_args()

	dataset = args.dataset
	topic = args.topic 	#dataset 1: paris, roller, venise,diving,timelapse, 
				   		#dataset 2: '0', '1', '2', '3', '4', '5', '6', '7', '8'
	fps = args.fps
	pred_window = fps

	if args.dataset != 1 and args.dataset != 2:
		print("Incorrect value of the Dataset ID provided!!...")
		print("======= EXIT ===========")
		exit()

	PATH = './Viewport/ds{}/'.format(dataset)
	if not os.path.exists(PATH):
		os.makedirs(PATH)
	
	#initialize head_orentiation
	print ("Extract Viewport for ds={}, topic={}".format(dataset, topic))
	dirpath1 = header.dirpath1	#u'./data/head-orientation/dataset1'
	dirpath2 = header.dirpath2	#u'./data/head-orientation/dataset2/Experiment_1'
	ext1 = header.ext1
	ext2 = header.ext2

	headoren = saldat_head_orientation.HeadOrientation(dirpath1, dirpath2, ext1, ext2)
	

	dirpath, filename_list, f_parse, f_extract_direction = headoren.load_filename_list(dataset, topic)
	series_ds_time = headoren.load_series_ds(filename_list, f_parse, dataset)
	_, vlength, _, _ = head_orientation_lib.topic_info_dict[topic]


	act_viewport_dict = {}
	pred_viewport_dict = {}

	cluster_user_data = []
	cluster_user_idx = []
	predict_user_data = []
	predict_user_idx = []

	print(np.array(series_ds_time).shape)

	# Change timestamp to frame number
	series_ds = []
	if(dataset == 1):
		for vector in series_ds_time:
			print(len(vector))
			viewport = []
			max_frame = int(vector[-1][0]*fps)
			for f in np.arange(0, max_frame):
				series_t = []
				series_Q = []
				for item in vector:
					if int(item[0]*fps) == f:
						series_t.append(item[0])
						series_Q.append([item[5], item[2], item[3], item[4]])

				if(len(series_t) != 0):
					mean_series_t = int(np.mean(np.array(series_t)) * fps)
					avg_Q = averageQuaternions(np.array(series_Q))
					viewport.append([mean_series_t, -1, avg_Q[0], avg_Q[1], avg_Q[2], avg_Q[3]])
			
			print("len(viewport): {}".format(len(viewport)))
			series_ds.append(viewport)

		print(np.array(series_ds).shape)
	else:
		for vector in series_ds_time:
			print(len(vector))
			viewport = []
			max_frame = int(vector[-1][0]*fps)
			for f in np.arange(0, max_frame):
				if f > 2100:
					break
				series_t = []
				series_Q = []
				for item in vector:
					if int(item[0]*fps) == f:
						series_t.append(item[0])
						series_Q.append([item[2], item[3], item[4], item[5]])

				if(len(series_t) != 0):
					mean_series_t = int(np.mean(np.array(series_t)) * fps)
					avg_Q = averageQuaternions(np.array(series_Q))
					viewport.append([mean_series_t, -1, avg_Q[0], avg_Q[1], avg_Q[2], avg_Q[3]])

			print("len(viewport): {}".format(len(viewport)))
			series_ds.append(viewport)

		print(np.array(series_ds).shape)

	pickle.dump(series_ds, open("series_ds_{}.pkl".format(topic), 'wb'))



	series_ds = pickle.load(open("series_ds_{}.pkl".format(topic), "rb"))
	
	# Remove users that does not have any data for more than prediction window number of frames
	remove_user = set()
	for usernum in range(len(series_ds)):
		prev_frame = None
		for frame in range(len(series_ds[usernum])):
			if frame == 0:
				prev_frame = series_ds[usernum][frame][0]
			else:
				new_frame = series_ds[usernum][frame][0]
				if new_frame - prev_frame >= pred_window:
					remove_user.add(usernum)
				prev_frame = new_frame
	
	for usernum in remove_user:
		del series_ds[usernum]



	last_frame_idx = [0 for i in range(len(series_ds))]
	last_frame = 0


	# Training and validation split
	for i in range(len(series_ds)):
		n = random.random()
		if n < 0.8:
			cluster_user_data.append(series_ds[i])
			cluster_user_idx.append(i)
		else:
			predict_user_data.append(series_ds[i])
			predict_user_idx.append(i)
			

	last_pred_view = [None] * len(predict_user_data)
	

	print("len(Cluster User Data): {}".format(len(cluster_user_data)))
	print("len(Predict User Data): {}".format(len(predict_user_data)))
	
	
	while(True):
		if last_frame == 2100:	# Break if the video frame number reaches this point (2100 for dataset 2, 1800 for dataset 1)
			break
			
		print("Last Time: {}".format(last_frame))		
		
		new_cluster_num = 0
		C_mean = {}	# key: cluster number, value: array of pred_window number of frames of quaternions
		C_mean_idx = {}
		Q = []

		C_mean_idx[new_cluster_num] = [cluster_user_idx[0]]
		C_mean[new_cluster_num] = []
		mean = [None for i in range(fps)]
		first_frame = cluster_user_data[0][0][0]
		
		
		# Initial Clustering to create the first cluster
		for i in range(len(cluster_user_data[0])):
			entry = cluster_user_data[0][i]
			if entry[0]-first_frame < pred_window:
				mean[entry[0] - first_frame] = [entry[2], entry[3], entry[4], entry[5]] # w,x,y,z
			else:
				last_frame_idx[cluster_user_idx[0]] = i
				break
		
		C_mean[new_cluster_num] = interpolate(mean)
		new_cluster_num += 1
		
		
		# For the rest of the series
		for i in range(len(series_ds)):
			if i in cluster_user_idx and i != cluster_user_idx[0]:
				flag = 0
				mean = [None for idx in range(fps)]
				
				# For the clusters already discovered
				for key, value in C_mean.items():
					num = tot = 0
					first_frame = series_ds[i][0][0]
					for j in range(len(series_ds[i])):
						entry = series_ds[i][j]
						if entry[0] < last_frame:
							first_frame = entry[0]
							continue
						if entry[0] - first_frame >= pred_window or entry[0] >= last_frame + pred_window or j == len(series_ds[i])-1:
							last_frame_idx[i] = j
							break
						
						# Calculate proximity of quaternions
						mean[entry[0] - first_frame] = [entry[2], entry[3], entry[4], entry[5]]	# w,x,y,z
						q1 = Quat(Quaternion.normalize([entry[3], entry[4], entry[5], entry[2]]))	# w,x,y,z -> x,y,z,w
						q2 = list(value[entry[0] - first_frame]) # w,x,y,z
						q2 = [q2[1], q2[2], q2[3], q2[0]]	# w,x,y,z -> x,y,z,w
						new_q = Quat(list(q2)).__mul__(q1.inv()).q	# x,y,z,w

						# Less than 30 degree
						if abs(2 * np.arccos(new_q[3])) < (np.pi/6):
							num += 1

						tot += 1
					

					# threshold 
					if tot is not 0 and float(num) / float(tot) > 0.9:
						mean = interpolate(mean)

						C_mean_idx[key].append(i)
						for i in range(len(value)):
							C_mean[key][i] = averageQuaternions(np.array([value[i], mean[i]]))
						flag = 1
						break

				# No cluster found, Create a new cluster
				if not flag:
					C_mean[new_cluster_num] = interpolate(mean)
					C_mean_idx[new_cluster_num] = [i]
					new_cluster_num += 1


		last_frame += pred_window
		print("Num of Clusters: {}".format(len(list(C_mean.keys()))))
		

		
		# Prediction 
		for i in range(len(predict_user_data)):
			print(i, end='\r')
			entry = predict_user_data[i]
			act_view = []
			pred_view = []
			mean = [None for idx in range(fps)]
			mean_act = [None for idx in range(fps)]
			flag = 0

			for key, value in C_mean.items():
				num = tot = 0
				first_frame = entry[0][0]

				for j in range(len(entry)):
					if entry[j][0] < last_frame - pred_window:
						first_frame = entry[j][0]
						continue
					if entry[j][0] >= last_frame or entry[j][0] - first_frame >= pred_window:
						break
					
					if not flag:
						# Get the viewport in vector form from vector_ds
						mean_act[entry[j][0] - first_frame] = [entry[j][2], entry[j][3], entry[j][4], entry[j][5]]


					# Caluclate proximity between two Quaternions
					mean[entry[j][0] - first_frame] = [entry[j][2], entry[j][3], entry[j][4], entry[j][5]]	# w,x,y,z
					q1 = Quat(Quaternion.normalize([entry[j][3], entry[j][4], entry[j][5], entry[j][2]]))	# x,y,z,w
					q2 = list(value[entry[j][0] - first_frame])	# w,x,y,z
					q2 = [q2[1], q2[2], q2[3], q2[0]]	# w,x,y,z -> x,y,z,w
					new_q = Quat(list(q2)).__mul__(q1.inv()).q	# x,y,z,w

					if 2 * np.arccos(new_q[3]) < (np.pi/6):
						num += 1

					tot += 1

				flag = 1
				if tot is not 0 and float(num) / float(tot) > 0.9:
					pred_quat = value	# w,x,y,z
					break
				else:
					if(last_frame - pred_window < pred_window):
						pred_quat = mean	# w,x,y,z	
					else:
						pred_quat = last_pred_view[i]
			
			last_pred_view[i] = interpolate(mean_act)
			
			act_quat = interpolate(mean_act)
			pred_quat = interpolate(pred_quat)
			
			
			# Convert the quaternion to tiles and store them
			for quat in act_quat:
				q = [quat[1], quat[2], quat[3], quat[0]]	# w,x,y,z -> x,y,z,w
				v = f_extract_direction(Quaternion.normalize(q))
				theta, phi = head_orientation_lib.vector_to_ang(v)
				x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)
				act_view.append((x,y))

			for quat in pred_quat:
				q = [quat[1], quat[2], quat[3], quat[0]]	# w,x,y,z -> x,y,z,w
				v = f_extract_direction(Quaternion.normalize(q))
				theta, phi = head_orientation_lib.vector_to_ang(v)
				x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)
				pred_view.append((x,y))

			if i in act_viewport_dict:
				act_viewport_dict[i].append(act_view)
			else:
				act_viewport_dict[i] = [act_view]

			if i in pred_viewport_dict:
				pred_viewport_dict[i].append(pred_view)
			else:
				pred_viewport_dict[i] = [pred_view]

		if last_frame >= int(vlength*fps):
			break
	
	
	pickle.dump(act_viewport_dict, open(PATH + 'act_viewport_ds{}_topic{}'.format(dataset, topic), 'wb'))
	pickle.dump(pred_viewport_dict, open(PATH + 'pred_viewport_ds{}_topic{}'.format(dataset, topic), 'wb'))
			

