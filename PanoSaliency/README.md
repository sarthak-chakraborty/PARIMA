# A Saliency Dataset for 360-Degree Videos 
This README file contains the instructions to use our 360-degree saliency dataset and how to reproduce the saliency maps which were discussed in the paper:

Anh Nguyen and Zhisheng Yan. A saliency dataset for 360-degree videos. In Proceedings of the 10th ACM Multimedia Systems Conference (MMSys’19), 2019.

The data and source code are distributed under the terms of the MIT license. Our contributions in this project are:
- A 360-degree saliency dataset with 50,654 saliency maps from 24 diverse videos. (original YouTube links of these videos are also provided).
- An open-source software to create 360-degree saliency maps from head tracking logs. 

To cite our paper, use this Bibtex code: 
```
@inproceedings{anguyen139,
AUTHOR = {Nguyen, Anh and Yan, Zhisheng},
 title = {A Saliency Dataset for 360-Degree Videos},
 booktitle = {Proceedings of the 10th ACM Multimedia Systems Conference (MMSys'19)},
 year = {2019}
}
```

# Paper Abstract
Despite the increasing popularity, realizing 360-degree videos in everyday applications is still challenging. Considering the unique viewing behavior in head-mounted display (HMD), understanding the saliency of 360-degree videos becomes the key to various 360-degree video research. Unfortunately, existing saliency datasets are either irrelevant to 360-degree videos or too small to support saliency modeling. In this paper, we introduce a large saliency dataset for 360-degree videos with 50,654 saliency maps from 24 diverse videos. The dataset is created by a new methodology supported by psychology studies in HMD viewing. Evaluation of the dataset shows that the generated saliency is highly correlated with the actual user fixation and that the saliency data can provide useful insight on user attention in 360-degree video viewing. The dataset and the program used to extract saliency are both made publicly available to facilitate future research. 

# 360-Degree Saliency Dataset  
The dataset includes 50,654 saliency maps from 24 videos. The saliency maps for each video are stored together in one file. The data in each file is organized into records. Each record has three fields: `timestamp`, `fixation`, and `saliency map`. The first field is the relative video time in seconds for the saliency maps. The second field is a list of fixation points. Each fixation point is a unit vector representing the head orientation in the three-dimensional space. The third field is the saliency map, where each pixel is a float number representing the saliency level in the original video frame.

To access the dataset, please follow the link provided inside `./data` folder.

# Program
## Program structure
`/data` contains the [link](https://zenodo.org/record/2641282#.XLYYGkMpDAg) to Zenodo.org where the saliency maps are stored.  
`/data/head-orientation` is the folder where input head tracking logs are supposed to reside. However, the input logs can also be specified inside the `/header.py` script file.   
`/get_fixation.py` is the main entry to create 360-degree saliency maps from head tracking logs.  
`/example.py`is the example Python code to retrieve the saliency maps from files in `data` folder.  

## Requirement & Installation
1. Download Python 2  
The program is developed Python 2.7. It is recommended that the [Anaconda2](https://www.anaconda.com/distribution/) packages is used  
2. Install [pyquarternion](http://kieranwynn.github.io/pyquaternion/).  
```sh
pip install pyquaternion
```
3. Collect Head Tracking Logs  
The program can received head tracking logs either in quarternion or Euler angles, and output saliency maps. Currently, head tracking logs are received from [Wu](https://wuchlei-thu.github.io/), [Corbillon](http://dash.ipv6.enstb.fr/headMovements/), and [Lo](https://nmsl.cs.nthu.edu.tw/360video/) . 

## Dataset Collection Program

To access our generated saliency maps, refer to the example in the file `./example.py`. The saliency maps are stored in python pickle format, which need to be extracted by a python program.  

To generate saliency map from heade tracking logs, refer to the file `./get_fixation.py`. The program assumes input head tracking logs have been downloaded and the file paths have been provided in `header.py`. To run the program, execute this command from terminal:  
```sh
python get_fixation.py <ds> <video>
```
The `get_fixation.py` file receives two parameters `ds` and `video`, which specify which head tracking logs and which video to convert. Here are the possible values:  
if `ds=1`, `video` can be `paris`, `roller`, `venise`, `diving`, `timelapse`  
if `ds=2`, `video` can be `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`  
if `ds=3`, `video` can be `coaster2_`, `coaster_`, `diving`, `drive`, `game`, `landscape`, `pacman`, `panel`, `ride`, `sport`  


## Edit
**`get_viewport.py` converts the head-tracking logs from quaternions to pixels on an equirectangular frame. Run this file with the above command line arguments to get viewport of all users for a particular video**

# License
This project is licensed under the terms of the MIT license.  

# Contact
If you have any general doubt about our work, please use the [public issues section](https://github.com/phananh1010/PanoSalNet/issues) on this github. Alternatively, drop us an e-mail at <mailto:anguyen139@student.gsu.edu> or <mailto:zyan@gsu.edu>.
