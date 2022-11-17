# PARIMA: Viewport Adaptive 360-Degree Video Streaming

This is the official repository corresponding to the paper titled **"PARIMA: Viewport Adaptive 360-Degree Video Streaming"** ([link](https://dl.acm.org/doi/10.1145/3442381.3450070)) accepted at the Proeedings of the 30th Web Conference 2021 (WWW '21), Ljubljana, Slovenia.

***Please cite our paper in any published work that uses any of these resources.***

```latex
@inproceedings{10.1145/3442381.3450070,  
author = {Chopra, Lovish and Chakraborty, Sarthak and Mondal, Abhijit and Chakraborty, Sandip},  
title = {PARIMA: Viewport Adaptive 360-Degree Video Streaming},  
year = {2021},  
isbn = {9781450383127},  
publisher = {Association for Computing Machinery},  
address = {New York, NY, USA},  
url = {https://doi.org/10.1145/3442381.3450070},  
doi = {10.1145/3442381.3450070},  
abstract = { With increasing advancements in technologies for capturing 360° videos, advances in streaming such videos have become a popular research topic. However, streaming 360° videos require high bandwidth, thus escalating the need for developing optimized streaming algorithms. Researchers have proposed various methods to tackle the problem, considering the network bandwidth or attempt to predict future viewports in advance. However, most of the existing works either (1) do not consider video contents to predict user viewport, or (2) do not adapt to user preferences dynamically, or (3) require a lot of training data for new videos, thus making them potentially unfit for video streaming purposes. We develop PARIMA, a fast and efficient online viewport prediction model that uses past viewports of users along with the trajectories of prime objects as a representative of video content to predict future viewports. We claim that the head movement of a user majorly depends upon the trajectories of the prime objects in the video. We employ a pyramid-based bitrate allocation scheme and perform a comprehensive evaluation of the performance of PARIMA. In our evaluation, we show that PARIMA outperforms state-of-the-art approaches, improving the Quality of Experience by over 30% while maintaining a short response time. },  
booktitle = {Proceedings of the Web Conference 2021},  
pages = {2379–2391},  
numpages = {13},  
keywords = {360° Video Streaming, Online Learning, Adaptive Streaming},  
location = {Ljubljana, Slovenia},  
series = {WWW '21}  
}
```


## Abstract
With increasing advancements in technologies for capturing 360-degree videos, advances in streaming such videos have become a popular research topic. However, streaming 360-degree videos require high bandwidth, thus escalating the need for developing optimized streaming algorithms. Researchers have proposed various methods to tackle the problem, considering the network bandwidth or attempt to predict future viewports in advance. However, most of the existing works either (1) do not consider video contents to predict user viewport, or (2) do not adapt to user preferences dynamically, or (3) require a lot of training data for new videos, thus making them potentially unfit for video streaming purposes. We develop *PARIMA*, a fast and efficient online viewport prediction model that uses past viewports of users along with the trajectories of prime objects as a representative of video content to predict future viewports. We claim that the head movement of a user majorly depends upon the trajectories of the prime objects in the video. We employ a pyramid-based bitrate allocation scheme and perform a comprehensive evaluation of the performance of *PARIMA*. In our evaluation, we show that *PARIMA* outperforms state-of-the-art approaches, improving the Quality of Experience by over 30% while maintaining a short response time.


## Folder Descriptions
		
		|
		|__ Baseline/          --> Contains the implementions of the baseline algorithms that we tested PARIMA against
		|   |
		|   |__ Clust/
		|   |__ NABA/
		|   |__ PanoSalnet/
		|
		|__ creme/              --> Modifications to the source code of the library to enable multi-frames prediction.
		|
		|__ PanoSaliency/       --> Contains the procedures to convert the head movement data from quaternion format to coordinates in an equirectangular frame
		|
		|__ Prediction/         --> Contains the implementation of PARIMA. Check 'Prediction/README.md` for further details.
		|
		|__ Preprocess/         --> Codes to perform the one-time video preprocessing at the Server end. Check `PreProcess/README.md` for further details
		|   |
		|   |__ FrameProjector/
		|   |__ YOLO/
		|   |__ ObjectTrack/
		|
	

## Requirements
Use `python3` for all the codes. Install the dependencies by running `pip install -r requirements.txt`.  

After the installation, source code for the package `creme` needs to be modified. For the same, go to the location(say, `PATH`) in your system where `creme` library source codes are stored(eg, `~/anaconda3/lib/python3.7/site-packages/creme/`) and copy the file `creme/linear_model/pa.py` into the appropriate subdirectory in `PATH`. 

**Note:** `creme` has now merged with `scikit-multiflow` to yield a new library ![`river`](https://github.com/online-ml/river). Appropriate changes will be needed in the code to make this code compatible.


## Datasets
In our experiments, we have particularly used two popular datasets containing several 360-degree videos of different categories along with head tracking log. 

1. Xavier Corbillon, Francesca De Simone, and Gwendal Simon. [2017]. *360-Degree Video Head Movement Dataset.* In Proceedings of the ACM MMSys 2017. 
It includes five videos freely viewed by 59 users each with each video watched for 70 seconds. The dataset is available from [http://dash.ipv6.enstb.fr/headMovements/](http://dash.ipv6.enstb.fr/headMovements/)
2. Chenglei Wu, Zhihao Tan, Zhi Wang, and Shiqiang Yang. [2017]. *A Dataset for Exploring User Behaviors in VR Spherical Video Streaming.* In Proceedings of the ACM MMSys 2017. 
It has nine popular videos watched by 48 users with an average view duration of 164 seconds. The dataset is available from [https://wuchlei-thu.github.io/](https://wuchlei-thu.github.io/)


## Pipeline
	Install Requirements and modify the source code (creme/)
					|
	Preprocess the video and store the object trajectories (Preprocess/)
					|
	Transform the quaternion format of the Head Movement Logs to Equirectangular form (PanoSaliency/)
					|
	Run PARIMA and calculate the QoE of a video (Prediction/)

**Note:** Follow the `README.md` inside each of the directories to get the details of the execution.
 

## License
The project is licensed under the terms of MIT License.
