import numpy as np 
import cv2 as cv
from PIL import Image 
import datetime


def initImage(width, height):
    image = np.ndarray((height, width, 3), dtype=np.uint8)
    return image


def saveImage(img, imgsize, destFile):
	mode = 'RGBA'
	arr = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
	if len(arr[0]) == 3:
		arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
	img =  Image.frombuffer(mode, imgsize, arr.tostring(), 'raw', mode, 0, 1)
	img.save(destFile)



image = Image.open('frame200.jpg')
image_val = image.load()
imsize = image.size
print(image.size)

print(imsize[0])

R = float(imsize[0])/(2*np.pi)
width, height = int(4*R)+1, int(4*R)+1

img1 = initImage(width, height)
img2 = initImage(width, height)
img3 = initImage(width, height)
img4 = initImage(width, height)


# for x in range(imsize[0]):
# 	for y in range(imsize[1]):
# 		u = float(x)/float(imsize[0])*360.0
# 		v = float(y)/float(imsize[1])*180.0

# 		longitude = u - 180.0
# 		latitude = v - 90.0

# 		theta = (np.pi*longitude)/180.0
# 		phi = (np.pi*latitude)/180.0

# 		x_e = R*np.cos(theta)*np.cos(phi)
# 		y_e = R*np.sin(theta)*np.cos(phi)
# 		z_e = R*np.sin(phi)


# 		if(theta >= 0):
# 			x_st = (2*R*x_e)/(R+y_e)
# 			y_st = (2*R*z_e)/(R+y_e)
# 			x_p = int(x_st + width/2)-1
# 			y_p = int(y_st + height/2)-1

# 			if(x_p>=width):
# 				x_p=width-1
# 			if(y_p>=height):
# 				y_p=height-1

# 			print(x_p, y_p, x_st, y_st, x, y)
# 			img1[y_p, x_p] = image_val[x,y]
# 		else:
# 			x_st = (2*R*x_e)/(R-y_e)
# 			y_st = (2*R*z_e)/(R-y_e)
# 			x_p = int(x_st + width/2)-1
# 			y_p = int(y_st + height/2)-1

# 			if(x_p>=width):
# 				x_p=width-1
# 			if(y_p>=height):
# 				y_p=height-1

# 			print(x_p, y_p, x_st, y_st, x, y)
			
# 			img2[y_p, x_p] = image_val[x,y]

# 		if(phi >= 0):
# 			x_st = (2*R*x_e)/(R+z_e)
# 			y_st = (2*R*y_e)/(R+z_e)
# 			x_p = int(x_st + width/2)-1
# 			y_p = int(y_st + height/2)-1

# 			if(x_p>=width):
# 				x_p=width-1
# 			if(y_p>=height):
# 				y_p=height-1

# 			print(x_p, y_p, x_st, y_st, x, y)
			
# 			img3[y_p, x_p] = image_val[x,y]
# 		else:
# 			x_st = (2*R*x_e)/(R-z_e)
# 			y_st = (2*R*y_e)/(R-z_e)
# 			x_p = int(x_st + width/2)-1
# 			y_p = int(y_st + height/2)-1

# 			if(x_p>=width):
# 				x_p=width-1
# 			if(y_p>=height):
# 				y_p=height-1

# 			print(x_p, y_p, x_st, y_st, x, y)
			
# 			img4[y_p, x_p] = image_val[x,y]



# saveImage(img1, (width,height), "img1_old.png")
# saveImage(img2, (width,height), "img2_old.png")
# saveImage(img3, (width,height), "img3_old.png")
# saveImage(img4, (width,height), "img4_old.png")


print(datetime.datetime.now())

for x in range(width):
	for y in range(height):
		u = x-(width-1)/2
		v = y-(height-1)/2

		if(u**2+v**2 <= 4*(R**2) or True):
			dem = u**2 + v**2 + 4*(R**2)

			# print(u,v)

			#img1
			x1, y1, z1 = (4*u*R**2)/dem, (4*R**3 - R*u**2 - R*v**2)/dem, (4*v*R**2)/dem
			theta, phi = np.arctan2(y1,x1), np.arcsin(z1/R)
			# assert(theta>=0)
			longitude, latitude = (180.0*theta)/np.pi, (180.0*phi)/np.pi

			x_e = int(((longitude+180.0)*imsize[0])/360.0)
			y_e = int(((latitude+90.0)*imsize[1])/180.0)

			if(x_e >= imsize[0]):
				x_e = imsize[0]-1
			if(y_e >= imsize[1]):
				y_e = imsize[1]-1

			img1[y,x] = image_val[x_e,y_e]



			#img2
			x1, y1, z1 = (4*u*R**2)/dem, (R*u**2 + R*v**2 - 4*R**3)/dem, (4*v*R**2)/dem
			theta, phi = np.arctan2(y1,x1), np.arcsin(z1/R)
			# assert(theta<0)
			longitude, latitude = (180.0*theta)/np.pi, (180.0*phi)/np.pi

			x_e = int(((longitude+180.0)*imsize[0])/360.0)
			y_e = int(((latitude+90.0)*imsize[1])/180.0)

			if(x_e >= imsize[0]):
				x_e = imsize[0]-1
			if(y_e >= imsize[1]):
				y_e = imsize[1]-1

			img2[y,x] = image_val[x_e,y_e]



			#img3
			x1, y1, z1 = (4*u*R**2)/dem, (4*v*R**2)/dem, (4*R**3 - R*u**2 - R*v**2)/dem
			theta, phi = np.arctan2(y1,x1), np.arcsin(z1/R)
			# assert(phi>=0)
			longitude, latitude = (180.0*theta)/np.pi, (180.0*phi)/np.pi

			x_e = int(((longitude+180.0)*imsize[0])/360.0)
			y_e = int(((latitude+90.0)*imsize[1])/180.0)

			if(x_e >= imsize[0]):
				x_e = imsize[0]-1
			if(y_e >= imsize[1]):
				y_e = imsize[1]-1

			img3[y,x] = image_val[x_e,y_e]



			#img4
			x1, y1, z1 = (4*u*R**2)/dem, (4*v*R**2)/dem, (R*u**2 + R*v**2 - 4*R**3)/dem
			theta, phi = np.arctan2(y1,x1), np.arcsin(z1/R)
			# assert(phi<0)
			longitude, latitude = (180.0*theta)/np.pi, (180.0*phi)/np.pi

			x_e = int(((longitude+180.0)*imsize[0])/360.0)
			y_e = int(((latitude+90.0)*imsize[1])/180.0)

			if(x_e >= imsize[0]):
				x_e = imsize[0]-1
			if(y_e >= imsize[1]):
				y_e = imsize[1]-1

			img4[y,x] = image_val[x_e,y_e]


print(datetime.datetime.now())

saveImage(img1, (width,height), "img1_eq.png")
saveImage(img2, (width,height), "img2_eq.png")
saveImage(img3, (width,height), "img3_eq.png")
saveImage(img4, (width,height), "img4_eq.png")
