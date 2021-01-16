## Viewport Prediction

### Requirements

	pip install creme 
	pip install statstools

Apart from the above two packages, common packages like `numpy`, `pandas` etc. are required.


### Steps
- `creme/` directory contains the modifications that we made to the source code of the library to enable multi-frames prediction. In your system, go to the location(say, `L`) where `creme` package is stored(eg, `~/anaconda3/lib/python3.7/site-packages/creme/`) and copy the file `creme/linear_model/pa.py` into the appropriate subdirectory in `L`. 
- For predicting the viewport and calculate qoe for a single user, change the `usernum` variable in `main.py` with the appropriate user number and run `python main.py <dataset> <topic> <fps> <offset for the start of the video> <preferred quality for the video to be played in>`.
- For predicting viewport of multiple users and calculate the average QoE, run `python obs_XXX.py <dataset> <topic> <fps> <offset for the start of the video> <preferred quality> <fps fraction>`
- `contribution.py` calculates the contribution(as percentage) of the objects and the previous viewport in predicting the next viewport 


### Note
Further polished codes with appropriate pipelining instructions will be made public later
