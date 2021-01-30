# PARIMA: Viewport Adaptive 360-degree Video Streaming


## Single User Execution
For predicting the viewport and calculate QoE for a single user, run `python main.py [...]`. The usage for the python file is as shown below:

	usage: main.py [-h] -D DATASET -T TOPIC --fps FPS [-O OFFSET] [-U USER] -Q
               QUALITY

	Run PARIMA algorithm and calculate QoE of a video for a single user

	optional arguments:
	  -h, --help            show this help message and exit
	  -D DATASET, --dataset DATASET
	                        Dataset ID (1 or 2)
	  -T TOPIC, --topic TOPIC
	                        Topic in the particular Dataset (video name)
	  --fps FPS             fps of the video
	  -O OFFSET, --offset OFFSET
	                        Offset for the start of the video in seconds (when the data was
	                        logged in the dataset) [default: 0]
	  -U USER, --user USER  User ID on which the algorithm will be run [default:
	                        0]
	  -Q QUALITY, --quality QUALITY
	                        Preferred bitrate quality of the video (360p, 480p,
	                        720p, 1080p, 1440p)


## Multiple User Execution
For predicting viewport of multiple users and calculate the average QoE, run `python obs_parima.py [...]`. The usage for the python file is as shown below:

	usage: obs_parima.py [-h] -D DATASET -T TOPIC --fps FPS [-O OFFSET]
                     [--fpsfrac FPSFRAC] -Q QUALITY

	Run PARIMA algorithm and calculate Average QoE of a video for all users

	optional arguments:
	  -h, --help            show this help message and exit
	  -D DATASET, --dataset DATASET
	                        Dataset ID (1 or 2)
	  -T TOPIC, --topic TOPIC
	                        Topic in the particular Dataset (video name)
	  --fps FPS             fps of the video
	  -O OFFSET, --offset OFFSET
	                        Offset for the start of the video in seconds (when the
	                        data was logged in the dataset) [default: 0]
	  --fpsfrac FPSFRAC     Fraction with which fps is to be multiplied to change
	                        the chunk size [default: 1.0]
	  -Q QUALITY, --quality QUALITY
	                        Preferred bitrate quality of the video (360p, 480p,
	                        720p, 1080p, 1440p)

The possible values for `dataset` and `topic`:  
if `ds=1`, `topic` can be `paris`, `roller`, `venise`, `diving`, `timelapse`  
if `ds=2`, `topic` can be `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`  

## Additional Inputs
Apart from the command line arguments, we will need to provide the path to the **Viewport Information** and the **Object Trajectory Information**.  

Paths to the same are defined in the `get_data()` function in the variables `VIEW_PATH` and `OBJ_PATH`


## Files Description
	- meta.json        : Specifies various constants and dataset related hyperparameters (eg., width and height of the video and number of users)
	- parima.py        : Implementation of PARIMA algorithm
	- bitrate.py       : Allocates bitrate to the tiles for each chunk
	- qoe.py           : Calculates the QoE of a video based on the allocated bitrate and the actual viewport
	- main.py          : Main function to run Single User Execution
	- obs_parima.py    : Main function to run Multiple User Execution

