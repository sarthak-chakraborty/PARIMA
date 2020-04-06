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


def generate(data):
	for i in data:
		yield i


fps=30
usernum=1
ncol_tiles=8
nrow_tiles=8
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
frame = int(view_info[-1][0]*fps*1.0/1000)
frame_nos.append(frame)
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



model = linear_model.PARegressor(C=1.0, mode=2, eps=0.1)

metric_X = metrics.MAE()
metric_Y = metrics.MAE()


out_data_X=[]
out_data_Y=[]
i=0
tile_manhattan_error=0

for inp,x,y in data:
	x_pred, y_pred = model.predict_one(inp)

	metric_X = metric_X.update(x, x_pred)
	metric_Y = metric_Y.update(y, y_pred)

	print("x_pred: "+str(x_pred))
	print("x: "+str(x))
	print(metric_X)
	print("y_pred: "+str(y_pred))
	print("y: "+str(y))
	print(metric_Y)
	
	actual_tile_col = int(x * ncol_tiles / 3840.0)
	actual_tile_row = int(y * nrow_tiles / 1920.0)
	pred_tile_col = int(x_pred * ncol_tiles / 3840.0)
	pred_tile_row = int(y_pred * nrow_tiles / 1920.0)

	print("("+str(actual_tile_col)+","+str(actual_tile_row)+"),("+str(pred_tile_col)+","+str(pred_tile_row)+")")
	
	tile_manhattan_error = tile_manhattan_error + abs(actual_tile_col-pred_tile_col)+abs(actual_tile_row-pred_tile_row)

	print("Manhattan Tile Error: "+str(tile_manhattan_error*1.0/(i+1)))	

	out_data_X.append([frame_nos[i], x, x_pred, metric_X.get()])
	out_data_Y.append([frame_nos[i], y, y_pred, metric_Y.get()])
	i += 1
	print("\n")
	model = model.fit_one(inp, x, y)
	

print(i)

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
	
out_data_X.append([frame_nos[-1], test_pred_x, test_actual_x, test_metric_X.get()])
out_data_Y.append([frame_nos[-1], test_pred_y, test_actual_y, test_metric_Y.get()])

np.save("Prediction_"+str(usernum)+"_x.npy",out_data_X)
np.save("Prediction_"+str(usernum)+"_y.npy",out_data_Y)