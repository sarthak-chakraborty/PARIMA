import argparse
import vrProjector2
import numpy as np 
import cv2 as cv
import os


def main():
	parser = argparse.ArgumentParser(description='Reproject bounding boxes')
	parser.add_argument('--source', required=True, help='Source Metadata File')
	parser.add_argument('--cubeMapDim', required=True, help='output images width in pixels')
	parser.add_argument('--dirPath', required=True, help='Path to directory of frames')
	args = parser.parse_args()
	

	out = vrProjector2.CubemapProjection()
	out.initImages(int(args.cubeMapDim), int(args.cubeMapDim))

	source = vrProjector2.EquirectangularProjection()
	source.loadImage(args.dirPath+"/frame0.jpg")
	out.reprojectToEquirectangular(args.source,source,int(args.cubeMapDim),int(args.cubeMapDim))

if __name__ == "__main__":
    main()