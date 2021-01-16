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
import naba as qoe
import matplotlib.pyplot as plt


# Dataset, topic, fps, offset, preferred quality
# fps fraction is the fraction of frames per second in one chunk (used for comparing qoe of multiple sizes). Keep it 1 for 1s chunk
dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
offset = int(sys.argv[4])
pref_quality = sys.argv[5]
fps_fraction = float(sys.argv[6])

nusers=0
final_qoe,qoe1,qoe2,qoe3,qoe4 = [],[],[],[],[]
save = './Predicted/QoE_Graphs/ds{}/'.format(dataset)

# Number of users
if dataset == 1:
	nusers=58
elif dataset == 2:
	nusers=48
else:
	nusers=4


nrow_tiles = 8
ncol_tiles = 8


for usernum in range(nusers):
	data, frame_nos = [],[]
	data, frame_nos, max_frame = get_data(data, frame_nos, dataset, topic, usernum)

	act_tiles, chunk_frames = tiling(data, frame_nos, max_frame)

	# To be consistent with our model
	i = 0
	while True:
		curr_frame=frame_nos[i]
		if curr_frame<5*fps:
			i += 1
		else:
			break

	frame_nos = frame_nos[i:]
	vid_bitrate = alloc_bitrate(frame_nos, chunk_frames, pref_quality)
	q, q1, q2, q3, q4 = calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames)
	final_qoe.append(q)
	qoe1.append(q1)
	qoe2.append(q2)
	qoe3.append(q3)
	qoe4.append(q4)
	print(q)

# Find averaged results
final_qoe.sort()
avg_qoe = np.mean(final_qoe)
avg_qoe1 = np.mean(qoe1)
avg_qoe2 = np.mean(qoe2)
avg_qoe3 = np.mean(qoe3)
avg_qoe4 = np.mean(qoe4)

# Print averaged results
print('Topic: '+topic)
print('Qoe NABA')
print('Pred nframe',(fps*fps_fraction))
print('Avg. QoE: ',avg_qoe,avg_qoe1,avg_qoe2,avg_qoe3,avg_qoe4)

print('\n\n')
