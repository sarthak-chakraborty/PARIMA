# PARIMA: Viewport Adaptive 360-Degree Video Streaming

This is the official repository corresponding to the paper titled "PARIMA: Viewport Adaptive 360-Degree Video Streaming"(link available soon) accepted at the Proeedings of the 30th Web Conference 2021 (WWW '21), Ljubljana, Slovenia.

***Please cite our paper in any published work that uses any of these resources.***

## Abstract

With increasing advancements in technologies for capturing 360-degree videos, advances in streaming such videos have become a popular research topic. However, streaming 360-degree videos require high bandwidth, thus escalating the need for developing optimized streaming algorithms. Researchers have proposed various methods to tackle the problem, considering the network bandwidth or attempt to predict future viewports in advance. However, most of the existing works either (1) do not consider video contents to predict user viewport, or (2) do not adapt to user preferences dynamically, or (3) require a lot of training data for new videos, thus making them potentially unfit for video streaming purposes. We develop *PARIMA*, a fast and efficient online viewport prediction model that uses past viewports of users along with the trajectories of prime objects as a representative of video content to predict future viewports. We claim that the head movement of a user majorly depends upon the trajectories of the prime objects in the video. We employ a pyramid-based bitrate allocation scheme and perform a comprehensive evaluation of the performance of *PARIMA*. In our evaluation, we show that *PARIMA* outperforms state-of-the-art approaches, improving the Quality of Experience by over 30% while maintaining a short response time.


## Folder Descriptions

- `video_preprocessing` contains codes for proprocessing the video content to obtain the object trajectories. 
- `PanoSaliency` contains the procedure to convert the head movement data from quaternion format to coordinates in an equirectangular frame
- `Prediction` contains the code to run our method of Viewport Prediction
- `Baseline` contains the codes for some of the baselines that we have used in our paper
