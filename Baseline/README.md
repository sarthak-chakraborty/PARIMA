# PARIMA: Viewport Adaptive 360-Degree Video Streaming

## Baseline Implementations
This directory contains the implementations of the various baselines that we have compared our algorithm against. Instructions for execution of each of the baselines are detailed as below:

### Clust
The Clust algorithm has been developed by A. Nasrabadi et. al. in their paper ["Viewport prediction for 360° videos: a clustering approach"](https://dl.acm.org/doi/abs/10.1145/3386290.3396934). 

To run:
	
	1. python clust.py [-h] -D DATASET -T TOPIC --fps FPS
	2. python qoe_clust.py [-h] -D DATASET -T TOPIC --fps FPS -Q QUALITY

The path to the viewport files can be mentioned in `header.py`. `clust.py` will store the viewports in the directory `./Viewport/ds1/` or `./Viewport/ds2/` which will automatically be created.


### NABA
NABA (Non-Adaptive Bitrate Allocation) allocates equal bitrates to all the tiles. To run:

	python obs_naba.py obs_naba.py [-h] -D DATASET -T TOPIC --fps FPS [-O OFFSET] [--fpsfrac FPSFRAC] -Q QUALITY

It will return the average QoE of all the users for a particular video as mentioned by `dataset` and `topic`.


### PanoSalNet

To, run PanoSalNet predictions on the videos, follow [https://github.com/phananh1010/PanoSalNet](https://github.com/phananh1010/PanoSalNet) and save the result in `./head_predictions/` directory. Variables `PATH_ACT` and `PATH_PRED` in `qoe_panosalent.py` denote the path for the actual viewports and the predicted saliency maps. 

The file `qoe_panosalnet.py` calculates the QoE of the videos whose predictions were given by PanoSalNet in the form of saliency maps. To calculate the QoE of the videos:

	python qoe_panosalnet.py [-h] -D DATASET -T TOPIC --fps FPS -Q QUALITY

It will return the average QoE of all the users for a particular video as mentioned by `dataset` and `topic`.
