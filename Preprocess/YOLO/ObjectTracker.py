import StringIO as io
import json
import time
import argparse
import numpy as np 
import cv2 as cv
import os
import math
# data = '{"filename":"results/result_00000002.jpg","tag":[{"person":32,"left":373,"right":515,"top":81,"bot":388},{"person":31,"left":439,"right":556,"top":65,"bot":384}]}'
# data = '{"filename":"results/result_00000056.jpg","tag":[{"car":28,"left":370,"right":719,"top":362,"bot":475}]}'

def getSphericalProj(imsize,R,x,y):
	u = float(x)/float(imsize[0])*360.0
	v = float(y)/float(imsize[1])*180.0

	longitude = u - 180.0
	latitude = v - 90.0

	theta = (np.pi*longitude)/180.0
	phi = (np.pi*latitude)/180.0

	x_e = R*np.cos(theta)*np.cos(phi)
	y_e = R*np.sin(theta)*np.cos(phi)
	z_e = R*np.sin(phi)
	return x_e,y_e,z_e

def getSolidAngle(imsize,R,x1,y1,x2,y2):
	xe1,ye1,ze1 = getSphericalProj(imsize,R,x1,y1)
	xe2,ye2,ze2 = getSphericalProj(imsize,R,x2,y2)

	diameter = math.sqrt((xe1-xe2)**2+(ye1-ye2)**2+(ze1-ze2)**2)
	angle = np.pi * diameter / (R**2)
	return angle


def same_Object(imsize,R,index,data1,data2,epsilon):

	angles = [[999 for j in range(len(data2))] for i in range(len(data1))]
	b_values = [[0 for j in range(len(data2))] for i in range(len(data1))]

	for i in range(len(data1)):
		for j in range(len(data2)):
			obj1=data1[i]
			obj2=data2[j]

			if obj1[1]==obj2[1] or obj1[-1]<0.9 or obj2[-1]<0.9:
				
				k = 0
				x1_sum=0
				x2_sum=0
				y1_sum=0
				y2_sum=0

				while k < len(obj1[2]):
					x1_sum=x1_sum+obj1[2][k]
					y1_sum=y1_sum+obj1[2][k+1]
					x2_sum=x2_sum+obj2[2][k]
					y2_sum=y2_sum+obj2[2][k+1]
					k=k+2

				x1_c=x1_sum*2/len(obj1[2])
				x2_c=x2_sum*2/len(obj1[2])
				y1_c=y1_sum*2/len(obj1[2])
				y2_c=y2_sum*2/len(obj1[2])

				angle=getSolidAngle(imsize,R,x1_c,y1_c,x2_c,y2_c)
				b_values[i][j]=(angle<epsilon)
				k = 0

				while k < len(obj1[2]):

					angle=getSolidAngle(imsize,R,obj1[2][k],obj1[2][k+1],obj2[2][k],obj2[2][k+1])
					print i,j,angle
					k=k+2
					angles[i][j]=min(angles[i][j],angle)

	obj_indices = {k:-1 for k in range(len(data2))}

	for i in range(len(data1)):
		minvalue=min(angles[i])
		j = angles[i].index(minvalue)
		i_j_same = 1

		if b_values[i][j] == 1:
			for k in range(len(data1)):
				if angles[k][j] < minvalue and b_values[k][j] == 1:
					i_j_same = 0
					break

			if i_j_same == 1:
				obj_indices[j]=i

	for j in obj_indices.keys():
		if obj_indices[j] == -1:
			data2[j][3] = index
			index = index + 1

		else:
			data2[j][3] = data1[obj_indices[j]][3]

	return data2,index


def track_Objects(imsize,R,line_arr):
	index = 0
	for i in range(len(line_arr[0])):
		line_arr[0][i][3] = index
		index = index+1
	for i in range(len(line_arr.keys())-1):
		print i
		line_arr[i+1],index = same_Object(imsize,R,index,line_arr[i],line_arr[i+1],0.001)
		#print line_arr[i]
		#print line_arr[i+1]
		#time.sleep(2)

	print index 

	return line_arr,index

def main():
	parser = argparse.ArgumentParser(description='Track objects')
	parser.add_argument('--sourceFile', required=True, help='Source File of Object Metadata')
	parser.add_argument('--imageDir', required=True, help='Image Directory')
	args = parser.parse_args()

	file = open(args.sourceFile, 'r')

	img = cv.imread(args.imageDir+"/frame0.jpg")

	imsize = [img.shape[1],img.shape[0]]
	R = float(imsize[0])/(2*np.pi)
	print "imsize = ",imsize,", R = ", R
	
	line_arr = {}
	for line in file:
		splitted=line[:-1].split(" ")
		frameno=int(splitted[0])
		objtype=splitted[1]
		i=2
		while i<len(splitted):
			try:
				coord=int(splitted[i])
				break
			except:
				objtype=objtype+"-"+splitted[i]
				i=i+1

		try:
			line_arr[frameno].append([frameno,objtype,[int(x) for x in splitted[i:-1]],-1,splitted[-1]])

		except:
			line_arr[frameno]=[[frameno,objtype,[int(x) for x in splitted[i:-1]],-1,splitted[-1]]]

	line_arr,index = track_Objects(imsize,R,line_arr)

	np.save(args.sourceFile[:-4]+"-tracked",np.array(line_arr))


	objects=[[None for i in range(len(line_arr))] for j in range(index)]

	for i in range(len(line_arr)):
		for y in line_arr[i]:
			objects[y[3]][i]=y[2]

	np.save(args.sourceFile[:-4]+"-object-info",np.array(objects))


if __name__ == "__main__":
    main()
