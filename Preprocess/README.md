## Video Pre-Processing

Run the following commands in sequence to generate Object tracking files

- Convert the equirectangular frame to its Cube Map projection

		cd FrameProjector 
		python vrProjectorWrapper.py --source <video file> --out 1024

	**Eg:** `python vrProjectorWrapper.py --source paris.mp4 --out 1024`
	**Output:** Folder containing the cubemap projection with name `paris`

- Stitch the frames obtained of a Cube Map projection

		python StitchingFrames.py --dirPath <directory containing cubemap projections and actual frames> --out 1024
		cd ..

	**Eg:** `python StitchingFrames.py --dirPath paris --out 1024`
	**Output:** Folder containing the stitched cubemap projection with name `paris_stitched`

- Run Object Detection

		cd YOLO
		python StitchedObjectDetectionWrapper.py --source <source directory with stitched images> --output <output filename>
		cd ..

	**Eg:** `python StitchedObjectDetectionWrapper.py --source ../FrameProjector/paris_stitched/ --output paris_obj.txt`
	**Output:** Object info in `paris_obj.txt`

- Reproject back into Equirectangular Frame

		cd FrameProjector
		python StitchedBoundingBoxConverter.py --source <sourceFileName> --cubeMapDim 1024 --dirPath <directory with frames>
		cd ..

	**Eg:** `python StitchedBoundingBoxConverter.py --source ../YOLO/paris_obj.txt --cubeMapDim 1024 --dirPath paris`
	**Output:** Equirectangular Projected Bounding Boxes in `../YOLO/paris_obj_equirectangular.txt`

- Run Object Tracking

		cd Object_tracking
		python tracker.py --sourceFile <sourceFileName> --dirPath <directory with frames> --outputnpy <outputfilename>
		cd ..

	**Eg:** `python tracker.py --sourceFile ../YOLO/paris_obj_equirectangular.txt --dirPath ../FrameProjector/paris --outputnpy paris_obj_traj.npy`
	**Output:** Object trajectories npy file
 