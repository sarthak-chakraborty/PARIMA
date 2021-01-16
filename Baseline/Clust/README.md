## Clust

Run `python clust.py <dataset> <topic> <fps>` to run the Clust algorithm as mentioned in ["Viewport prediction for 360° videos: a clustering approach"](https://dl.acm.org/doi/abs/10.1145/3386290.3396934). The path to the viewport files can be mentioned in `header.py`. 

The output pickle files for the predicted viewports along with the actual viewports can then be processed for QoE calculation. `python qoe_clust.py <dataset> <topic>` can be run with correct file paths to obtain the QoE.