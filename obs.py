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
import qoe_ts as qoe
import matplotlib.pyplot as plt



dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
offset = int(sys.argv[4])
pref_quality = sys.argv[5]
nusers=0
manhattan_error, x_mae, y_mae, final_qoe = [],[],[],[]
matrix_error = []
save = './Predicted/QoE_Graphs/ds{}/'.format(dataset)

if dataset == 1:
	nusers=58
elif dataset == 2:
	nusers=48
else:
	nusers=4


nrow_tiles = 8
ncol_tiles = 8


for usernum in range(nusers):
	print('User_{}'.format(usernum))
	user_matrix_error = 0.

	data, frame_nos = [],[]
	data, frame_nos, max_frame, total_objects = qoe.get_data(data, frame_nos, dataset, topic, usernum+1)

	act_tiles,pred_tiles,chunk_frames, err, x, y = qoe.build_model(data, frame_nos, max_frame, total_objects)
	manhattan_error.append(err)
	x_mae.append(x)
	y_mae.append(y)

	i = 0
	while True:
		curr_frame=frame_nos[i]
		if curr_frame<5*fps:
			i += 1
		else:
			break

	for fr in range(len(pred_tiles)):
		act_tile = act_tiles[fr]
		pred_tile = pred_tiles[fr]
		act_prob = np.array([[0. for i in range(ncol_tiles)] for j in range(nrow_tiles)])
		pred_prob = np.array([[0. for i in range(ncol_tiles)] for j in range(nrow_tiles)])
		act_prob[act_tile[0]][act_tile[1]] = 1

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
	vid_bitrate = qoe.alloc_bitrate(pred_tiles, frame_nos, chunk_frames, pref_quality)
	q = qoe.calc_qoe(vid_bitrate, act_tiles, frame_nos, chunk_frames)
	final_qoe.append(q)


avg_qoe = np.mean(final_qoe)
avg_manhattan_error = np.mean(list(zip(*manhattan_error)), axis=1)
avg_matrix_error = np.mean(matrix_error)
avg_x_mae = np.mean(list(zip(*x_mae)), axis=1)
avg_y_mae = np.mean(list(zip(*y_mae)), axis=1)

print('Topic: '+topic)
print('Avg. QoE: {}'.format(avg_qoe))
print('Avg. Manhattan Error: {}'.format(avg_manhattan_error[-1]))
print('Avg. Matrix Error: {}'.format(avg_matrix_error))
print('Avg. X_MAE: {}'.format(avg_x_mae[-1]))
print('Avg. Y_MAE: {}'.format(avg_y_mae[-1]))
print('\n\n')


# plt.figure()
# plt.plot(avg_manhattan_error, marker='.', color='b')
# plt.xlabel('Chunks')
# plt.ylabel('Manhattan Tile Error')
# plt.savefig(save + 'Manhattan/topic{}.png'.format(topic))

# plt.figure()
# plt.plot(avg_x_mae, marker='.', color='r')
# plt.xlabel('Chunks')
# plt.ylabel('X_MAE')
# plt.savefig(save + 'X_MAE/topic{}.png'.format(topic))

# plt.figure()
# plt.plot(avg_y_mae, marker='.', color='g')
# plt.xlabel('Chunks')
# plt.ylabel('Y_MAE')
# plt.savefig(save + 'Y_MAE/topic{}.png'.format(topic))
