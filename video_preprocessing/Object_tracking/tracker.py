from centroidtracker.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2 as cv
import argparse 

parser = argparse.ArgumentParser(description='Track objects')
parser.add_argument('--sourceFile', required=True, help='Source File of Object Metadata')
parser.add_argument('--dirPath', required=True, help='Image Directory')
parser.add_argument('--outputnpy',required=True,help='Output file path')
args = parser.parse_args()

img = cv.imread(args.dirPath+"/frame0.jpg")
imsize = [img.shape[1],img.shape[0]]
R = float(imsize[0])/(2*np.pi)
print "imsize = ",imsize,", R = ", R

f_in=open(args.sourceFile,"r")
objects=f_in.readlines()

total_frames=int(objects[-1].split(" ")[0])+1
print total_frames
object_dict={i:[] for i in range(total_frames)}

for x in objects:
	y=x.split(" ")
	frameno=int(y[0])
	#print frameno,
	objtype=y[1]
	i=2
	while 1:
		try:
			x1=int(y[i])
			break
		except:
			objtype=objtype+" "+y[i]
			i=i+1

	x1=int(y[i])
	y1=int(y[i+1])
	x2=int(y[i+2])
	y2=int(y[i+3])
	x3=int(y[i+4])
	y3=int(y[i+5])
	x4=int(y[i+6])
	y4=int(y[i+7])
	confidence = float(y[i+8])
	if confidence>0.5:
		object_dict[frameno].append((x1,y1,x2,y2,x3,y3,x4,y4))

#print object_dict

object_trajectories = {i:{} for i in range(total_frames)}
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker(imsize,R)
(H, W) = (None, None)
max_id=0
for i in range(total_frames):
	objects=ct.update(object_dict[i],i)
	for (objectID,centroid) in objects.items():
		max_id=max(max_id,objectID)
		object_trajectories[i][objectID] = centroid
		#print centroid[0],centroid[1]
		#text = "ID {}".format(objectID)
		#cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		#cv2.circle(frame, (centroid[0], centroid[1]), 50, (255,0,0), 6)


print max_id
missingFrames = ct.getMissing()
intervals = ct.getInterval(total_frames)


for i in range(max_id+1):
	missing = missingFrames[i]
	k=0
	while k<len(missing):
		contiguous = [missing[k]]

		while k < len(missing):
			try:
				if missing[k] + 1 == missing[k+1]:
					k=k+1
					contiguous.append(missing[k])
				else:
					break

			except:
				break
		k=k+1
		
		first_frame = contiguous[0]-1
		last_frame = contiguous[-1]+1
		if last_frame >= total_frames:
			continue
			
		pos_first = object_trajectories[first_frame][i]
		pos_last = object_trajectories[last_frame][i]
		fraction = 1.0/(len(contiguous)+1)
		change_x = fraction*(pos_last[0]-pos_first[0])
		change_y = fraction*(pos_last[1]-pos_first[1])

		for l in range(len(contiguous)):
			object_trajectories[contiguous[l]][i] = (pos_first[0]+(l+1)*change_x,pos_first[1]+(l+1)*change_y)


np.save(args.outputnpy, object_trajectories) 

