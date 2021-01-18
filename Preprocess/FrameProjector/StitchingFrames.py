import cv2 as cv
import numpy as np 
import argparse
import os

def main():
	parser = argparse.ArgumentParser(description='Stitch Multiple Frames')
	parser.add_argument('--out', required=True, help='cubemap image height and width in pixels')
	parser.add_argument('--dirPath', required=True, help='Path to directory of frames')
	args=parser.parse_args()
	
	args.out=int(args.out)
	args.out=int(args.out)
	path, dirs, files = next(os.walk(args.dirPath))
	file_count = len(files)
	img=np.zeros((args.out*3,args.out*4,3))

	new_dir=args.dirPath+"_stitched"
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

	n_frames=file_count/7
	print(n_frames)
	for i in range(n_frames):
		if not os.path.exists(new_dir+"/"+str(i)+".png"):
			
			img_left=cv.imread(args.dirPath+"/"+str(i)+"_left.png")
			img_right=cv.imread(args.dirPath+"/"+str(i)+"_right.png")
			img_front=cv.imread(args.dirPath+"/"+str(i)+"_front.png")
			img_back=cv.imread(args.dirPath+"/"+str(i)+"_back.png")
			img_top=cv.imread(args.dirPath+"/"+str(i)+"_top.png")
			img_bottom=cv.imread(args.dirPath+"/"+str(i)+"_bottom.png")

			img_bottom=cv.rotate(img_bottom,0)
			img_top=cv.rotate(img_top,1)

			img[args.out:2*args.out,0:args.out]=img_left
			img[args.out:2*args.out,args.out:2*args.out]=img_front
			img[args.out:2*args.out,2*args.out:3*args.out]=img_right
			img[args.out:2*args.out,3*args.out:4*args.out]=img_back
			img[0:args.out,3*args.out:4*args.out]=img_top
			img[2*args.out:3*args.out,0:args.out]=img_bottom
			cv.imwrite(new_dir+"/"+str(i)+".png",img)

if __name__ == "__main__":
	main()