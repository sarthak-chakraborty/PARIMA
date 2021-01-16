import os
import numpy as np
import pickle
import header
import sys


import head_orientation_lib
import saldat_head_orientation
import saldat_saliency

if __name__ == "__main__":

	#specify dataset & video name to extract
	TOPIC = sys.argv[2]#for 6, modify 2 places, for loop vlength, and output file with _part
	DELTA = 0.06
	fps = int(sys.argv[3])
	
	dataset = int(sys.argv[1])#saldat_head_orientation.HeadOrientation._DATASET2
	topic = TOPIC#dataset 1: paris, roller, venise,diving,timelapse, 
				   #dataset 2: '0', '1', '2', '3', '4', '5', '6', '7', '8'
				   #dataset 3: ['coaster2_', 'coaster_', 'diving', 'drive', 'game', 'landscape', 'pacman', 'panel', 'ride', 'sport']
	#specify output address to store the saliency maps
	
	#initialize head_oren
	print ("generating saliency maps for ds={}, topic={}".format(dataset, TOPIC))
	dirpath1 = header.dirpath1#u'./data/head-orientation/dataset1'
	dirpath2 = header.dirpath2#u'./data/head-orientation/dataset2/Experiment_1'
	dirpath3 = header.dirpath3#u'./data/head-orientation/dataset3/sensory/orientation'
	ext1 = header.ext1
	ext2 = header.ext2
	ext3 = header.ext3
	headoren = saldat_head_orientation.HeadOrientation(dirpath1, dirpath2, dirpath3, ext1, ext2, ext3)
	#initialize 
	var = 20
	salsal = saldat_saliency.Fixation(var)
	
	dirpath, filename_list, f_parse, f_extract_direction = headoren.load_filename_list(dataset, topic)
	series_ds = headoren.load_series_ds(filename_list, f_parse, dataset)
	vector_ds = headoren.headpos_to_headvec(series_ds, f_extract_direction, dataset)
	# vector_ds = headoren.cutoff_vel_acc(vector_ds, dataset=dataset)

	_, vlength, _, _ = head_orientation_lib.topic_info_dict[topic]
	_bp=2
	_ap=1

	if(dataset == 1):
		user = 1
		for vector in vector_ds:
			viewport = []
			print("Usernum={}".format(user))
			max_frame = vector[-1][0]
			for f in np.arange(0, max_frame):
				dt = 1.0/fps
				series_t = []
				series_v = []
				print(f, end='\r')
				for item in vector:
					if item[0] == f:
						series_t.append(item[0])
						series_v.append(item[1])

				if(len(series_t) != 0):
					mean_series_t = np.mean(np.array(series_t)) * dt
					mean_series_v = np.mean(np.array(series_v), axis=0)

					theta, phi = head_orientation_lib.vector_to_ang(mean_series_v)
					x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)

					viewport.append([mean_series_t, (x,y)])

			pickle.dump(viewport, open('./data/ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, user), 'wb'))
			user += 1
	else:
		user = 1
		for vector in vector_ds:
			viewport = []
			print("Usernum={}".format(user))
			dt = fps
			for item in vector:
				print(item[0], end='\r')
				mean_series_t = item[0]
				mean_series_v = np.mean(np.array([item[1]]), axis=0)

				theta, phi = head_orientation_lib.vector_to_ang(mean_series_v)
				x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)

				viewport.append([mean_series_t, (x,y)])

			pickle.dump(viewport, open('./data/ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, user), 'wb'))
			user += 1



			# for t in np.arange(0, vlength, DELTA):
			# 	dt = 1.0/30
			# 	series_t = []
			# 	series_v = []
			# 	print(t, end='\r')
			# 	for item in vector:
			# 		if item[0] >= t - _bp*dt and item[0] <= t + _ap*dt:
			# 			series_t.append(item[0])
			# 			series_v.append(item[1])

			# 	if(len(series_t) != 0):
			# 		mean_series_t = np.mean(np.array(series_t))
			# 		mean_series_v = np.mean(np.array(series_v), axis=0)

			# 		theta, phi = head_orientation_lib.vector_to_ang(mean_series_v)
			# 		x, y = head_orientation_lib.ang_to_geoxy(theta, phi, head_orientation_lib.H, head_orientation_lib.W)

			# 		viewport.append([mean_series_t, (x,y)])
