## PanoSalNet

The file `qoe_panosalnet.py` calculates the QoE of the videos whose predictions were given by PanoSalNet in the form of saliency maps. 

To, run PanoSalNet predictions on the videos, follow [https://github.com/phananh1010/PanoSalNet](https://github.com/phananh1010/PanoSalNet) and save the result in `head_predictions/` directory located in the same directory as `qoe_panosalnet.py`. Variable `path_act` and `path_pred` denote the path for the actual viewports and the predicted saliency maps. 

Run `python qoe_panosalnet.py <dataset> <topic> <fps> <preferred quality>` to get the QoE value.