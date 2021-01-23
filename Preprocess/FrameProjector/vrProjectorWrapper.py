
import argparse
import vrProjector
import numpy as np 
import cv2 as cv
import os

def main():
	parser = argparse.ArgumentParser(description='Reproject photospheres')
	parser.add_argument('--source', required=True, help='Source Video')
	parser.add_argument('--out', required=True, help='output images width and height in pixels')
	args = parser.parse_args()

	if not os.path.exists(args.source[:-4]):
		os.mkdir(args.source[:-4])
	os.chdir(args.source[:-4])

	pixels=[]

	out = vrProjector.CubemapProjection()
	out.initImages(int(args.out), int(args.out))

	video = cv.VideoCapture("../"+args.source)
	source = vrProjector.EquirectangularProjection()

	success,image = video.read()
	count = 0
	dim=(0,0)

	while success:
		if count == 0:
			width =  image.shape[1]
			height = image.shape[0]
			print("Width: {}, Height: {}".format(width, height))
			if width > 2 * height:
				dim = (2 * height, height)
			else:
				if width < 2 * height:
					dim = (width, width / 2)
		image = cv.resize(image,dim, interpolation = cv.INTER_AREA)

		cv.imwrite("frame%d.jpg" % count, image)
		source.loadImage("frame%d.jpg" % count)
		out.reprojectToThis(source)
		out.saveImages("%d_front.png" % count, "%d_right.png" % count, "%d_back.png" % count, "%d_left.png" % count, "%d_top.png" % count, "%d_bottom.png" % count)
		success,image = video.read()
		count += 1

if __name__ == "__main__":
    main()