# PARIMA: Viewport Adaptive 360-Degree Video Streaming

## Viewport Extraction from Head Movement Logs

The code in this section is based on the works *Anh Nguyen and Zhishen Yan. A saliency dataset for 360-degree videos. In Proceedings of the 10th ACM Multimedia Systems Conference (MMSys’19), 2019.*
<br/>
The official github repository for the above work is [https://github.com/phananh1010/PanoSaliency](https://github.com/phananh1010/PanoSaliency).

The code in this section contain certain modifications to the original repository where we extract just the viewport information on an equirectangular frame from the head movement logs. We don't compute the saliency maps from the head-tracking logs.


## Program structure
	/data -> Contains youtube links for the videos used for viewport extraction.  
	/data/head-orientation -> Directory where input head tracking logs are supposed to reside. However, the input logs can also be specified inside the `/header.py` script file.   
	/get_viewport.py -> Main entry to create viewport files from head tracking logs.  


## Execution Steps
To generate viewports from heade tracking logs, refer to the file `./get_viewport.py`. The program assumes input head tracking logs have been downloaded and the file paths have been provided in `header.py`. It converts the head-tracking logs from quaternions to pixels on an equirectangular frame. To run the file, follow the usage as shown by `python get_viewport.py --help`
	
	usage: get_viewport.py [-h] -D DATASET -T TOPIC --fps FPS

	Run Viewport Extraction Algorithm

	optional arguments:
	  -h, --help            show this help message and exit
	  -D DATASET, --dataset DATASET
	                        Dataset ID (1 or 2)
	  -T TOPIC, --topic TOPIC
	                        Topic in the particular Dataset (video name)
	  --fps FPS             fps of the video

Here are the possible values for `dataset` and `topic`:  
if `ds=1`, `topic` can be `paris`, `roller`, `venise`, `diving`, `timelapse`  
if `ds=2`, `topic` can be `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`  
