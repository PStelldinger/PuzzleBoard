# About

The PuzzleBoard is an alternative calibration pattern for geometric camera calibration. The often used checkerboard calibration pattern (based on Zhang et al.'s seminal work) lacks any positional encoding. Thus, the calibration pattern must be completely visible without any occlusions. In some applications alternative solutions are preferred. E.g. ArUco or ChArUco boards allow partial occlusions, but require a higher camera resolution to read the code.

The PuzzleBoard is a new alternative that combines the advantages of checkerboard calibration patterns with a lightweight position coding that can be decoded at very low resolutions. Even very small image sections are unique in terms of translation and rotation. The decoding algorithm includes error correction and is computationally efficient.

PuzzleBoards can be used not only for camera calibration but also for camera pose estimation and marker-based object localization tasks (the markers will then be small subpatterns of the PuzzleBoard pattern).

This reposiory gives a first implementation of the decoding algorithm for pictures taken from a PuzzleBoard target.

If you want to print a PuzzleBoard target you can use the PuzzleBoard generator from the following website:

https://users.informatik.haw-hamburg.de/~stelldinger/pub/PuzzleBoard/welcome.html

there you will also find demo videos and further information about the PuzzleBoard.

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
