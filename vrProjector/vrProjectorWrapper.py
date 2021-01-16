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
			print(width,height)
			if width > 2*height:
				dim = (2*height,height)
			else:
				if width < 2*height:
					dim = (width, int(width/2))
		image = cv.resize(image,dim,interpolation = cv.INTER_AREA)

		cv.imwrite("frame%d.jpg" % count, image)
		source.loadImage("frame%d.jpg" % count)
		out.reprojectToThis(source)
		out.saveImages("%d_front.png"%count, "%d_right.png"%count, "%d_back.png"%count, "%d_left.png"%count, "%d_top.png"%count, "%d_bottom.png"%count)
		success,image = video.read()
		count += 1

if __name__ == "__main__":
    main()
