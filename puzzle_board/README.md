# Usage

## Needed pip packages
1. opencv2
2. numpy
3. sklearn

## Structure
`example.py` is a simple example program calling `puzzleboard_detector.py` for a given image and plotting the detections.
`puzzleboard_detector.py` is the main entrance point of this project. 
Running this with an rgb uint8 image as parameter will run the entire pipeline from initial corner detection to MST and Grid Decoding. 
It returns the discrete grid coordinates as well as the subpixel image positions of the found grid points.
