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

fps=30
usernum=1
ncol_tiles=8
nrow_tiles=8
pred_nframe=30

obj_info = np.load('lovish_stitched_metadata-equirectangular-object-info.npy', allow_pickle=True, encoding='latin1')
view_info = np.load('Viewport'+str(usernum)+'.npy', allow_pickle=True, encoding='latin1')
data=[]
frame_nos=[]


for i in range(len(view_info)-1):
	frame = int(view_info[i][0]*1.0*fps/1000)
	frame_nos.append(frame)
	if(frame >= 577):
		break
	X={}
	X['VIEWPORT_x']=int(view_info[i][1][0]*3840.0/1280)
	X['VIEWPORT_y']=int(view_info[i][1][1]*1920.0/720)
	for j in range(len(obj_info)):
		if obj_info[j][frame] == None:
			X['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
			X['OBJ_'+str(j)+'_y']=np.random.normal(0,1)
		else:
			centroid_x = float(obj_info[j][frame][0]+obj_info[j][frame][2]+obj_info[j][frame][4]+obj_info[j][frame][6])/4
			centroid_y = float(obj_info[j][frame][1]+obj_info[j][frame][3]+obj_info[j][frame][5]+obj_info[j][frame][7])/4
			X['OBJ_'+str(j)+'_x']=centroid_x
			X['OBJ_'+str(j)+'_y']=centroid_y

	data.append((X, int(view_info[i+1][1][0]*3840.0/1280), int(view_info[i+1][1][1]*1920.0/720)))


test={}
testframe = int(view_info[-1][0]*fps*1.0/1000)
frame_nos.append(testframe)
if(frame >= 577):
	frame = 576
test['VIEWPORT_x']=int(view_info[-1][1][0]*3840.0/1280)
test['VIEWPORT_y']=int(view_info[-1][1][1]*1920.0/720)
for j in range(len(obj_info)):
	if obj_info[j][frame] == None:
		test['OBJ_'+str(j)+'_x']=np.random.normal(0,1)
		test['OBJ_'+str(j)+'_y']=np.random.normal(0,1)
	else:
		centroid_x = float(obj_info[j][frame][0]+obj_info[j][frame][2]+obj_info[j][frame][4]+obj_info[j][frame][6])/4
		centroid_y = float(obj_info[j][frame][1]+obj_info[j][frame][3]+obj_info[j][frame][5]+obj_info[j][frame][7])/4
		test['OBJ_'+str(j)+'_x']=centroid_x
		test['OBJ_'+str(j)+'_y']=centroid_y



model = linear_model.PARegressor(C=1.0, mode=2, eps=0.1, data=data)
metric_X = metrics.MAE()
metric_Y = metrics.MAE()


out_data_X=[]
out_data_Y=[]
i=0
manhattan_error = 0
tile_manhattan_error=0
count=0

while True:
	print(i)
	curr_frame = frame_nos[i]
	# if(curr_frame <= 5):
	# 	frames = [i]
	# 	for k in range(i+1,len(frame_nos)):
	# 		if(frame_nos[k] < curr_frame + 5):
	# 			frames.append(k)
	# 		else:
	# 			i=k
	# 			break

	# 	model = model.fit_n(frames)
	# 	continue


	nframe = min(pred_nframe, 576 - frame_nos[i])
	if(nframe <= 0):
		break

	frames = [i]
	for k in range(i+1,len(frame_nos)):
		if(frame_nos[k] < curr_frame + nframe):
			frames.append(k)
		else:
			i=k
			break

	x_pred, y_pred = 0,0
	for k in range(len(frames)):
		[inp_k, x_act, y_act] = data[frames[k]]
		if(k == 0):
			x_pred, y_pred = model.predict_one(inp_k)
		else:
			inp_k['VIEWPORT_x'] = x_pred
			inp_k['VIEWPORT_y'] = y_pred
			x_pred, y_pred = model.predict_one(inp_k)

		# print("x_pred: "+str(x_pred))
		# print("x: "+str(x_act))
		# print("y_pred: "+str(y_pred))
		# print("y: "+str(y_act))	

		metric_X = metric_X.update(x_act, x_pred)
		metric_Y = metric_Y.update(y_act, y_pred)

		actual_tile_col = int(x_act * ncol_tiles / 3840.0)
		actual_tile_row = int(y_act * nrow_tiles / 1920.0)
		pred_tile_col = int(x_pred * ncol_tiles / 3840.0)
		pred_tile_row = int(y_pred * nrow_tiles / 1920.0)

		# print(abs(actual_tile_col - pred_tile_col) + abs(actual_tile_row - pred_tile_row))
		# print("("+str(actual_tile_col)+","+str(actual_tile_row)+"),("+str(pred_tile_col)+","+str(pred_tile_row)+")")

		tile_manhattan_error += abs(actual_tile_col - pred_tile_col) + abs(actual_tile_row - pred_tile_row)

		out_data_X.append([frame_nos[frames[k]], x_act, x_pred, metric_X.get()])
		out_data_Y.append([frame_nos[frames[k]], y_act, y_pred, metric_Y.get()])

		count = count+1


	# inp_pred = inp
	# for k in frames:
	# 	x_pred, y_pred = model.predict_one(inp_pred)

	# 	if(cur_frame+k in frame_nos):
	# 		ind = frame_nos.index(cur_frame+k)
	# 		x_act = data[ind][1]
	# 		y_act = data[ind][2]
	# 		metric_X = metric_X.update(x_act, x_pred)
	# 		metric_Y = metric_Y.update(y_act, y_pred)

	# 		actual_tile_col = int(x_act * ncol_tiles / 3840.0)
	# 		actual_tile_row = int(y_act * nrow_tiles / 1920.0)
	# 		pred_tile_col = int(x_pred * ncol_tiles / 3840.0)
	# 		pred_tile_row = int(y_pred * nrow_tiles / 1920.0)

	# 		manhattan_error += abs(actual_tile_col-pred_tile_col) + abs(actual_tile_row-pred_tile_row)
		
	# 	inp_pred['VIEWPORT_x'] = x_pred
	# 	inp_pred['VIEWPORT_y'] = y_pred
	# 	x_pred_old = x_pred
	# 	y_pred_old = y_pred

	# tile_manhattan_error = float(tile_manhattan_error)/count


	# print("x_pred: "+str(x_pred))
	# print("x: "+str(x))
	print(metric_X)
	# print("y_pred: "+str(y_pred))
	# print("y: "+str(y))
	print(metric_Y)
	
	# actual_tile_col = int(x * ncol_tiles / 3840.0)
	# actual_tile_row = int(y * nrow_tiles / 1920.0)
	# pred_tile_col = int(x_pred * ncol_tiles / 3840.0)
	# pred_tile_row = int(y_pred * nrow_tiles / 1920.0)

	# print("("+str(actual_tile_col)+","+str(actual_tile_row)+"),("+str(pred_tile_col)+","+str(pred_tile_row)+")")
	
	# tile_manhattan_error = tile_manhattan_error + abs(actual_tile_col-pred_tile_col) + abs(actual_tile_row-pred_tile_row)

	print("Manhattan Tile Error: "+str(tile_manhattan_error*1.0 / count))	

	# out_data_X.append([frame_nos[i], x, x_pred, metric_X.get()])
	# out_data_Y.append([frame_nos[i], y, y_pred, metric_Y.get()])
	# i += 1
	print("\n")
	model = model.fit_n(frames)



print("\nPrediction:")
test_pred_x, test_pred_y = model.predict_one(test)
test_actual_x = int(view_info[-1][1][0]*3840.0/1280)
test_actual_y = int(view_info[-1][1][1]*1920.0/720)

test_metric_X = metric_X.update(test_actual_x, test_pred_x)
test_metric_Y = metric_Y.update(test_actual_y, test_pred_y)

print(test_pred_x)
print(test_actual_x)
print(test_metric_X)
print(test_pred_y)
print(test_actual_y)
print(test_metric_Y)

actual_tile_col=(int(test_actual_x*8/3840))
actual_tile_row=(int(test_actual_y*8/1920))
pred_tile_col=(int(test_pred_x*8/3840))
pred_tile_row=(int(test_pred_y*8/1920))

print("("+str(actual_tile_col)+","+str(actual_tile_row)+"),("+str(pred_tile_col)+","+str(pred_tile_row)+")")
	


out_data_X.append([frame_nos[-1], test_actual_x, test_pred_x, test_metric_X.get()])
out_data_Y.append([frame_nos[-1], test_actual_y, test_pred_y, test_metric_Y.get()])

np.save("Prediction_"+str(usernum)+"_x.npy",out_data_X)
np.save("Prediction_"+str(usernum)+"_y.npy",out_data_Y)