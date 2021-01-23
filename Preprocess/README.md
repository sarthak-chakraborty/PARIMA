# PARIMA: Viewport Adaptive 360-degree Video Streaming

## Video Pre-Processing

Run the following commands in sequence to generate Object tracking files

1. Convert the equirectangular frame to its Cube Map projection

		cd FrameProjector 
		python vrProjectorWrapper.py --source <video file> --out 1024

	**Eg:** `python vrProjectorWrapper.py --source paris.mp4 --out 1024`
	<br/>
	**Output:** Directory named `./paris/` containing the cubemap projection

2. Stitch the frames obtained of a Cube Map projection

		python StitchingFrames.py --dirPath <directory containing cubemap projections> --out 1024
		cd ..

	**Eg:** `python StitchingFrames.py --dirPath paris --out 1024`
	<br/>
	**Output:** Directory named `./paris_stitched/` containing the stitched cubemap projection

3. Run Object Detection

		cd YOLO
		python StitchedObjectDetectionWrapper.py --source <source directory with stitched images> --output <output filename>
		cd ..

	**Eg:** `python StitchedObjectDetectionWrapper.py --source ../FrameProjector/paris_stitched/ --output paris_obj.txt`
	<br/>
	**Output:** Object information in `./paris_obj.txt`

4. Reproject back into Equirectangular Frame

		cd FrameProjector
		python StitchedBoundingBoxConverter.py --source <sourceFileName> --cubeMapDim 1024 --dirPath <directory with frames>
		cd ..

	**Eg:** `python StitchedBoundingBoxConverter.py --source ../YOLO/paris_obj.txt --cubeMapDim 1024 --dirPath paris`
	<br/>
	**Output:** Equirectangular Projected Bounding Boxes in `../YOLO/paris_obj_equirectangular.txt`

5. Run Object Tracking

		cd ObjectTrack
		python tracker.py --sourceFile <sourceFileName> --dirPath <directory with frames> --outputnpy <outputfilename>
		cd ..

	**Eg:** `python tracker.py --sourceFile ../YOLO/paris_obj_equirectangular.txt --dirPath ../FrameProjector/paris --outputnpy paris_obj_traj.npy`
	<br/>
	**Output:** Object trajectories in `../../Obj_traj/paris_obj_traj.npy`
 
