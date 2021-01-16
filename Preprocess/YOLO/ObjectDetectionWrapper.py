
import argparse
import os

views=["_front.png","_back.png","_top.png","_bottom.png","_left.png","_right.png"]

parser = argparse.ArgumentParser()
curpath=os.getcwd()
parser.add_argument('--source', required=True, help='The directory where the cubemap projection of all frames are.')
parser.add_argument('--metafilename',required=True,help='The file where metadata will be stored')
args=parser.parse_args()
path, dirs, files = next(os.walk(args.source))
file_count = len(files)

n_frames=file_count/7

for i in range(n_frames):
	for v in views:
		filename=str(i)+v
		position=v[1:-4]
		print i
		os.system("python yolo.py --image-path="+args.source+"/"+filename+" --storefilename "+args.metafilename+" --framenum "+str(i)+" --position "+position)



