# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import math 

class CentroidTracker():
	def __init__(self, imsize,R, maxDisappeared=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.imsize = imsize
		self.R = R
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.missing = {}
		self.interval = {}
		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def getMissing(self):
		return self.missing

	def getInterval(self,total_frames):
		interval_corrected = self.interval
		for i in interval_corrected.keys():
			if interval_corrected[i][1] == 0:
				interval_corrected[i]=(interval_corrected[i][0],total_frames-1)
				if self.missing[i]!=[]:
					if self.missing[i][-1] == total_frames-1:
						for j in range(1,total_frames):
							try:
								self.missing[i].remove(total_frames-j)
							except:
								break

		return interval_corrected

	def getSphericalProj(self,x,y):
		u = float(x)/float(self.imsize[0])*360.0
		v = float(y)/float(self.imsize[1])*180.0

		longitude = u - 180.0
		latitude = v - 90.0

		theta = (np.pi*longitude)/180.0
		phi = (np.pi*latitude)/180.0

		x_e = self.R*np.cos(theta)*np.cos(phi)
		y_e = self.R*np.sin(theta)*np.cos(phi)
		z_e = self.R*np.sin(phi)
		return x_e,y_e,z_e

	def spherical_distance(self,object1,object2):
		xe1,ye1,ze1 = self.getSphericalProj(object1[0],object1[1])
		xe2,ye2,ze2 = self.getSphericalProj(object2[0],object2[1])

		diameter = math.sqrt((xe1-xe2)**2+(ye1-ye2)**2+(ze1-ze2)**2)
		angle = np.pi * diameter / (self.R**2)
		return angle

	def distance_matrix(self,objectCentroids,inputCentroids):
		n_objects = len(objectCentroids)
		n_inputs = len(inputCentroids)

		D = np.empty((n_objects,n_inputs))

		for i in range(n_objects):
			for j in range(n_inputs):
				D[i][j] = self.spherical_distance(objectCentroids[i],inputCentroids[j])

		return D

	def register(self, centroid,frameno):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.missing[self.nextObjectID] = []
		self.interval[self.nextObjectID] = (frameno,0)
		self.nextObjectID += 1
		

	def deregister(self, objectID,frameno):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]
		self.interval[objectID] = (self.interval[objectID][0],frameno-self.maxDisappeared-1)
		self.missing[objectID] = self.missing[objectID][:-50]

	def update(self, rects,frameno):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID,frameno)
				else:
					try:
						self.missing[objectID].append(frameno)
					except:
						self.missing[objectID]=[frameno]

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (x1,y1,x2,y2,x3,y3,x4,y4)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((x1+x2+x3+x4) / 4.0)
			cY = int((y1+y2+y3+y4) / 4.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i],frameno)

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = self.distance_matrix(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1


					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID,frameno)
					else:
						try:
							self.missing[objectID].append(frameno)
						except:
							self.missing[objectID]=[frameno]

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col],frameno)

		# return the set of trackable objects
		return self.objects