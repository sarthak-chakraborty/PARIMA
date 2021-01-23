# Copyright 2016 Bhautik J Joshi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .AbstractProjection import AbstractProjection
from PIL import Image
import math
from datetime import datetime

class CubemapProjection(AbstractProjection):
	def __init__(self):
		AbstractProjection.__init__(self)

	def set_angular_resolution(self):
		# imsize on a face: covers 90 degrees
		#     |\
		#  0.5| \
		#     |  \
		#     -----
		#     1/self.imsize[0]
		#  angular res ~= arctan(1/self.imsize[0], 0.5)
		self.angular_resolution = math.atan2(1/self.imsize[0], 0.5)

	def loadImages(self, front, right, back, left, top, bottom):
		self.front, self.imsize = self._loadImage(front)
		self.right, self.imsize = self._loadImage(right)
		self.back, self.imsize = self._loadImage(back)
		self.left, self.imsize = self._loadImage(left)
		self.top, self.imsize = self._loadImage(top)
		self.bottom, self.imsize = self._loadImage(bottom)
		self.set_angular_resolution()

	def initImages(self, width, height):
		self.imsize = (width, height)
		self.front = self._initImage(width, height)
		self.right = self._initImage(width, height)
		self.back = self._initImage(width, height)
		self.left = self._initImage(width, height)
		self.top = self._initImage(width, height)
		self.bottom = self._initImage(width, height)
		self.set_angular_resolution()

	def saveImages(self, front, right, back, left, top, bottom):
		self._saveImage(self.front, self.imsize, front)
		self._saveImage(self.right, self.imsize, right)
		self._saveImage(self.back, self.imsize, back)
		self._saveImage(self.left, self.imsize, left)
		self._saveImage(self.top, self.imsize, top)
		self._saveImage(self.bottom, self.imsize, bottom)

	def _pixel_value(self, angle):
		theta = angle[0]
		phi = angle[1]
		if theta is None or phi is None:
		  return (0,0,0)

		sphere_pnt = self.point_on_sphere(theta, phi)
		x = sphere_pnt[0]
		y = sphere_pnt[1]
		z = sphere_pnt[2]

		eps = 1e-6

		if math.fabs(x)>eps:
			if x>0:
				t = 0.5/x
				u = 0.5+t*y
				v = 0.5+t*z
				if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
				  return self.get_pixel_from_uv(u, v, self.front)
			elif x<0:
				t = 0.5/-x
				u = 0.5+t*-y
				v = 0.5+t*z
				if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
				  return self.get_pixel_from_uv(u, v, self.back)

		if math.fabs(y)>eps:
			if y>0:
				t = 0.5/y
				u = 0.5+t*-x
				v = 0.5+t*z
				if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
				  return self.get_pixel_from_uv(u, v, self.right)
			elif y<0:
				t = 0.5/-y
				u = 0.5+t*x
				v = 0.5+t*z
				if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
				  return self.get_pixel_from_uv(u, v, self.left)

		if math.fabs(z)>eps:
			if z>0:
				t = 0.5/z
				u = 0.5+t*y
				v = 0.5+t*-x
				if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
				  return self.get_pixel_from_uv(u, v, self.bottom)
			elif z<0:
				t = 0.5/-z
				u = 0.5+t*y
				v = 0.5+t*x
				if u>=0.0 and u<=1.0 and v>=0.0 and v<=1.0:
				  return self.get_pixel_from_uv(u, v, self.top)

		return None

	def get_theta_phi(self, _x, _y, _z):
		dv = math.sqrt(_x*_x + _y*_y + _z*_z)
		x = _x/dv
		y = _y/dv
		z = _z/dv
		theta = math.atan2(y, x)
		phi = math.asin(z)
		return theta, phi

	@staticmethod
	def angular_position(texcoord):
		u = texcoord[0]
		v = texcoord[1]
		return None

	def reprojectToThis(self, sourceProjection):
		halfcubeedge = 1.0

		curr=datetime.now()
		if len(self.image_theta_phi)==0:
			for x in range(self.imsize[0]):
				for y in range(self.imsize[1]):
					u = 2.0*(float(x)/float(self.imsize[0])-0.5)
					v = 2.0*(float(y)/float(self.imsize[1])-0.5)

					# front
					theta, phi = self.get_theta_phi(halfcubeedge, u, v)
					x1,y1,self.front[y,x] = sourceProjection.pixel_value((theta, phi))
					self.image_theta_phi.append((x1,y1))

					# right
					theta, phi = self.get_theta_phi(-u, halfcubeedge, v)
					x1,y1,self.right[y,x] = sourceProjection.pixel_value((theta, phi))
					self.image_theta_phi.append((x1,y1))

					# left
					theta, phi = self.get_theta_phi(u, -halfcubeedge, v)
					x1,y1,self.left[y,x] = sourceProjection.pixel_value((theta, phi))
					self.image_theta_phi.append((x1,y1))

					# back
					theta, phi = self.get_theta_phi(-halfcubeedge, -u, v)
					x1,y1,self.back[y,x] = sourceProjection.pixel_value((theta, phi))
					self.image_theta_phi.append((x1,y1))

					# bottom
					theta, phi = self.get_theta_phi(-v, u, halfcubeedge)
					x1,y1,self.bottom[y,x] = sourceProjection.pixel_value((theta, phi))
					self.image_theta_phi.append((x1,y1))

					# top
					theta, phi = self.get_theta_phi(v, u, -halfcubeedge)
					x1,y1,self.top[y,x] = sourceProjection.pixel_value((theta, phi))
					self.image_theta_phi.append((x1,y1))

		else:
			i=0
			for x in range(self.imsize[0]):
				for y in range(self.imsize[1]):
					self.front[y,x] = sourceProjection.pixel_value_from_xy(self.image_theta_phi[i])
					self.right[y,x] = sourceProjection.pixel_value_from_xy(self.image_theta_phi[i+1])
					self.left[y,x] = sourceProjection.pixel_value_from_xy(self.image_theta_phi[i+2])
					self.back[y,x] = sourceProjection.pixel_value_from_xy(self.image_theta_phi[i+3])
					self.bottom[y,x] = sourceProjection.pixel_value_from_xy(self.image_theta_phi[i+4])
					self.top[y,x] = sourceProjection.pixel_value_from_xy(self.image_theta_phi[i+5])
					i=i+6

	def findpart(self,x,y,width,height):
		if x<width and y>=height and y<2*height:
			return x,y-height,'left'
		elif x<width and y>=2*height:
			return width-(y-2*height),x,'bottom'
		elif x>=width and x<2*width and y>=height and y<2*height:
			return x-width,y-height,'front'
		elif x>=2*width and x<3*width and y>=height and y<2*height:
			return x-2*width,y-height,'right'
		elif x>=3*width and y>=height and y<2*height:
			return x-3*width,y-height,'back'
		elif x>=3*width and y<height:
			return width-(x-3*width),height-y,'top'
		else:
			if x<2*width:
				return width,2*height,'left'
			else:
				return 3*width,height,'back'

	def reprojectToEquirectangular(self, filepath, sourceProjection, width, height):
		f_in=open(filepath,"r")
		f_out=open(filepath[:-4]+"-equirectangular.txt","w")
		objects=f_in.readlines()
		halfcubeedge = 1.0

		for x in objects:
			y=x.split(" ")
			frameno=y[0]
			print(frameno),
			objtype=y[2]
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
			confidence=y[i+4]
			x1eq=0
			y1eq=0
			x2eq=0
			y2eq=0

			sides=[]

			sides.append(self.findpart(x1,y1,width,height))
			sides.append(self.findpart(x2,y1,width,height))
			sides.append(self.findpart(x1,y2,width,height))
			sides.append(self.findpart(x2,y2,width,height))

			f_out.write(frameno+" "+objtype+" ")
			print(sides)
			for s in sides: 
				print(s)

				side=s[2]
				x1=s[0]
				y1=s[1]
				u1 = 2.0*(float(x1)/float(self.imsize[0])-0.5)
				v1 = 2.0*(float(y1)/float(self.imsize[1])-0.5)

				if side=='front':
					theta, phi = self.get_theta_phi(halfcubeedge, u1, v1)
					x1eq,y1eq,_ = sourceProjection.pixel_value((theta, phi))

				elif side=='back':
					theta, phi = self.get_theta_phi(-halfcubeedge, -u1, v1)
					x1eq,y1eq,_ = sourceProjection.pixel_value((theta, phi))

				elif side=='top':
					theta, phi = self.get_theta_phi(v1, u1, -halfcubeedge)
					x1eq,y1eq,_ = sourceProjection.pixel_value((theta, phi))

				elif side=='bottom':
					theta, phi = self.get_theta_phi(-v1, u1, halfcubeedge)
					x1eq,y1eq,_ = sourceProjection.pixel_value((theta, phi))

				elif side=='left':
					theta, phi = self.get_theta_phi(u1, -halfcubeedge, v1)
					x1eq,y1eq,_ = sourceProjection.pixel_value((theta, phi))

				elif side=='right':
					theta, phi = self.get_theta_phi(-u1, halfcubeedge, v1)
					x1eq,y1eq,_ = sourceProjection.pixel_value((theta, phi))

				else:
					print("ERROR")

				f_out.write(str(x1eq)+" "+str(y1eq)+" ")

			f_out.write(str(confidence))

		f_in.close()
		f_out.close()

