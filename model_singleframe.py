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


dataset = int(sys.argv[1])
topic = sys.argv[2]
fps=int(sys.argv[3])
offset = int(sys.argv[4])

usernum=1
ncol_tiles=8
nrow_tiles=8
frame_nos = []
data = []

# # OUR DATA
# width=3840.0
# height=1920.0
# view_width = 1280
# view_height = 720
# milisec = 1000.0

# ds 1
width=3840.0
height=1920.0
view_width = 3840.0
view_height = 2048.0
milisec = 1.0



# # FOR OWN DATA
# obj_info = np.load('lovish_stitched_object_trajectory_converted.npy', allow_pickle=True).item()
# view_info = np.load('Viewport'+str(usernum)+'.npy', allow_pickle=True, encoding='latin1')

obj_info = np.load('Obj_traj/ds{}/ds{}_topic{}.npy'.format(dataset, dataset, topic), allow_pickle=True,  encoding='latin1').item()
view_info = pickle.load(open('Viewport/ds{}/viewport_ds{}_topic{}_user{}'.format(dataset, dataset, topic, usernum), 'rb'), encoding='latin1')


n_objects = []
for i in obj_info.keys():
	try:
		n_objects.append(max(obj_info[i].keys()))
	except:
		n_objects.append(0)
total_objects=max(n_objects)
max_frame = int(view_info[-1][0]*1.0*fps/milisec)

for i in range(len(view_info)-1):
	frame = int(view_info[i][0]*1.0*fps/milisec)
	frame += int(offset*1.0*fps/milisec)

	frame_nos.append(frame)
	if(frame > max_frame):
		break
	X={}
	X['VIEWPORT_x']=int(view_info[i][1][0]*width/view_width)
	X['VIEWPORT_y']=int(view_info[i][1][1]*height/view_height)
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


	data.append((X, int(view_info[i+1][1][0]*width/view_width),int(view_info[i+1][1][1]*height/view_height)))

test={}
testframe = int(view_info[-1][0]*fps*1.0/milisec)
testframe += int(offset*1.0*fps/milisec)
frame_nos.append(testframe)
if(testframe > max_frame):
	testframe = max_frame-1
test['VIEWPORT_x']=int(view_info[-1][1][0]*width/view_width)
test['VIEWPORT_y']=int(view_info[-1][1][1]*height/view_height)
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



model = linear_model.PARegressor(C=1.0, mode=2, eps=0.1)

metric_X = metrics.MAE()
metric_Y = metrics.MAE()


out_data_X=[]
out_data_Y=[]
i=0
tile_manhattan_error=0

for inp,x,y in data:
	x_pred, y_pred = model.predict_one(inp, False, None, None)

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