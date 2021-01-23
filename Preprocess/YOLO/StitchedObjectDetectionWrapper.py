
import argparse
import os

parser = argparse.ArgumentParser()
curpath=os.getcwd()
parser.add_argument('--source', required=True, help='The directory where the stitched projection of all frames are.')
parser.add_argument('--output',required=True,help='The file where metadata will be stored')
args=parser.parse_args()

path, dirs, files = next(os.walk(args.source))
file_count = len(files)

for i in range(file_count):
	filename = str(i) + ".png"
	print(i)
	os.system("python yolo.py --image-path=" + args.source + "/" + filename + " --storefilename " + args.output + " --framenum " + str(i))

