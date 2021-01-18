# PARIMA: Viewport Adaptive 360-degree Video Streaming

## YOLOv3-Object-Detection-with-OpenCV

The reporsitory [https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV](https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV) has been used to write the code for YOLO Object Detection Model. 
<br/>
This project implements an image and video object detection classifier using pretrained yolov3 models. 
The yolov3 models are taken from the official yolov3 paper which was released in 2018. The yolov3 implementation is from [darknet](https://github.com/pjreddie/darknet).


## Steps

1. To infer on an image that is stored on your local machine
```
python yolo.py --image-path='/path/to/image/' --storefilename='/path/to/store'
```
2. To infer on a video that is stored on your local machine
```
python yolo.py --video-path='/path/to/video/'
```



Note: This works considering you have the `weights` and `config` files at the yolov3-coco directory.
<br/>
If the files are located somewhere else then mention the path while calling the `yolo.py`. For more details
```
python yolo.py --help
```

## References

1. [PyImageSearch YOLOv3 Object Detection with OpenCV Blog](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)

