import argparse
import vrProjector
import numpy as np 
import cv2 as cv
import os


def main():
	parser = argparse.ArgumentParser(description='Reproject bounding boxes')
	parser.add_argument('--sourceMetaFilePath', required=True, help='Source Metadata File')
	parser.add_argument('--outWidth', required=True, help='output images width in pixels')
	parser.add_argument('--outHeight', required=True, help='output images height in pixels')
	parser.add_argument('--dirPath', required=True, help='Path to directory of frames')
	args = parser.parse_args()
	

	out = vrProjector.CubemapProjection()
	out.initImages(int(args.outWidth), int(args.outHeight))

	source = vrProjector.EquirectangularProjection()
	source.loadImage(args.dirPath + "/frame0.jpg")
	out.reprojectToEquirectangular(args.sourceMetaFilePath, source)

if __name__ == "__main__":
    main()
