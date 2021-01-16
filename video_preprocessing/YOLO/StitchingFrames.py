import cv2 as cv
import numpy as np 
import argparse
import os

def main():
	parser = argparse.ArgumentParser(description='Stitch Multiple Frames')
	parser.add_argument('--outWidth', required=True, help='output images width in pixels')
	parser.add_argument('--outHeight', required=True, help='output images height in pixels')
	parser.add_argument('--dirPath', required=True, help='Path to directory of frames')
	args=parser.parse_args()
	args.outHeight=int(args.outHeight)
	args.outWidth=int(args.outWidth)
	path, dirs, files = next(os.walk(args.dirPath))
	file_count = len(files)
	img=np.zeros((args.outHeight*3,args.outWidth*4,3))

	new_dir=args.dirPath[:-1]+"_stitched"
	os.mkdir(new_dir)

	n_frames=file_count/7
	print n_frames
	for i in range(n_frames):
		img_left=cv.imread(args.dirPath+str(i)+"_left.png")
		img_right=cv.imread(args.dirPath+str(i)+"_right.png")
		img_front=cv.imread(args.dirPath+str(i)+"_front.png")
		img_back=cv.imread(args.dirPath+str(i)+"_back.png")
		img_top=cv.imread(args.dirPath+str(i)+"_top.png")
		img_bottom=cv.imread(args.dirPath+str(i)+"_bottom.png")

		img_bottom=cv.rotate(img_bottom,0)
		img_top=cv.rotate(img_top,1)

		img[args.outHeight:2*args.outHeight,0:args.outWidth]=img_left
		img[args.outHeight:2*args.outHeight,args.outWidth:2*args.outWidth]=img_front
		img[args.outHeight:2*args.outHeight,2*args.outWidth:3*args.outWidth]=img_right
		img[args.outHeight:2*args.outHeight,3*args.outWidth:4*args.outWidth]=img_back
		img[0:args.outHeight,3*args.outWidth:4*args.outWidth]=img_top
		img[2*args.outHeight:3*args.outHeight,0:args.outWidth]=img_bottom

		cv.imwrite(new_dir+"/"+str(i)+".png",img)

if __name__ == "__main__":
	main()