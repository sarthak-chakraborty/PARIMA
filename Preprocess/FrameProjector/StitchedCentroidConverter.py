import argparse
import vrProjector
import numpy as np 
import cv2 as cv
import os


def main():
	parser = argparse.ArgumentParser(description='Reproject bounding boxes')
	parser.add_argument('--source', required=True, help='Source Metadata File')
	parser.add_argument('--cubeMapDim', required=True, help='cubemap images width in pixels')
	parser.add_argument('--dirPath', required=True, help='directory with actual frames')
	parser.add_argument('--output', required=True, help='Output file name')
	args = parser.parse_args()

	out = vrProjector.CubemapProjection()
	out.initImages(int(args.cubeMapDim), int(args.cubeMapDim))

	source = vrProjector.EquirectangularProjection()
	source.loadImage(args.dirPath + "/frame0.jpg")
	out.reprojectToEquirectangular(args.source, source, int(args.cubeMapDim), int(args.cubeMapDim), args.output)

if __name__ == "__main__":
    main()